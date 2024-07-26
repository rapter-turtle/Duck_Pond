#include <ros/ros.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <Eigen/Dense>
#include <std_msgs/Float64MultiArray.h> //include the lib
#include <tf/tf.h>
#include <cmath>
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

/* Parameters */
#define UTM_ZON 52
#define Wa 6378137
#define Wb 6356752.314245
#define We 0.081819190842965
#define Weps 0.006739496742333
#define PI 3.1415926535897931
#define DEF_DEG2RAD 3.1415926535897931 / 180.0
#define DEF_RAD2DEG 180.0 / 3.1415926535897931
#define DEF_BIAS_HDG 0.
#define nx 8 
#define ngps 2
#define nimu 4
#define npn 3 // n process noise
#define DEF_BIAS_accx 0.0
#define DEF_BIAS_accy 0.0

// x, y, psi, u, v, r, ax, ay
// gps: x, y
// imu: psi, r, ax, ay

const double pi = 3.14159265358979323846;
const double a = 6378137.0; // WGS84 major axis
const double f = 1 / 298.257223563; // WGS84 flattening
const double k0 = 0.9996; // UTM scale factor
const float IMU_HZ = 15;

class SensorFusionEKF {
public:
    SensorFusionEKF() {
        imu_sub_ = nh_.subscribe("/imu/data", 1, &SensorFusionEKF::imuCallback, this);
        gps_sub_ = nh_.subscribe("/ublox_gps/fix", 1, &SensorFusionEKF::gpsCallback, this);
        state_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/ekf/estimated_state", 1);
    
        // Initialize the state and covariance matrices
        x_ = VectorXd(nx);
        P_ = MatrixXd(nx, nx);
        A_ = MatrixXd(nx, nx);
        C_ = MatrixXd(ngps+nimu, nx);
        K_ = MatrixXd(ngps+nimu, ngps+nimu);
        IKC_ = MatrixXd(nx, nx);
        E_ = MatrixXd(nx, npn);
        Q_ = MatrixXd(npn, npn);
        R_imu_ = MatrixXd(nimu, nimu);
        R_gps_ = MatrixXd(ngps, ngps);
        R_ = MatrixXd(ngps+nimu, ngps+nimu);

        // initial state
        x_ << 0, 0, 0, 0, 0, 0, 0, 0;

        // initial covariance matrix
        P_ << 1, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 0, 0, 1;
        P_ = P_ * 1e1;

        // measurement matrix
        C_ << 1, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 0, 0, 1;

        // process noise - r, ax, ay
        Q_ << 0.001, 0, 0,
              0, 0.0001, 0,
              0, 0, 0.0001;

        R_gps_ << 0.00001, 0,
                  0, 0.00001;
        R_imu_ << 0.00000001, 0, 0, 0,
                  0, 0.00000001, 0, 0,
                  0, 0, 0.00000001, 0,
                  0, 0, 0, 0.00000001;

        R_.setZero();
        E_.setZero();
        R_.block<2,2>(0,0) = R_gps_;
        R_.block<4,4>(2,2) = R_imu_;

        z_imu_linear_acc = VectorXd(2);
        z_imu_ang_vel = VectorXd(1);
    }

    double calpi2pi(double dPI)
    {
        // Assume that dPI is deg (only for IMU yaw bias)
        if (dPI >= PI)
            dPI = dPI - (2.0 * PI);
        else
            dPI = dPI;

        if (dPI <= ((-1.0) * PI))
            dPI = dPI + (2.0 * PI);
        else
            dPI = dPI;

        return dPI;
    }
    void imuCallback_ang(const geometry_msgs::Vector3Stamped::ConstPtr& msg) {
        z_imu_ang_vel << msg->vector.z;
        }
    void imuCallback_linacc(const geometry_msgs::Vector3Stamped::ConstPtr& msg) {
        z_imu_linear_acc << msg->vector.x-DEF_BIAS_accx, msg->vector.y-DEF_BIAS_accy;
        }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        if (start_EKF){
        tf::Quaternion q(
            msg->orientation.x,
            msg->orientation.y,
            msg->orientation.z,
            msg->orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        yaw = yaw + DEF_BIAS_HDG; // declination
        yaw = calpi2pi(yaw);
        //for raw data plot

        // IMU data (acceleration / yaw rate)
        z_imu = VectorXd(4);
        z_imu << yaw, z_imu_ang_vel[0], z_imu_linear_acc[0], z_imu_linear_acc[1];

        // double dt = (msg->header.stamp - last_time_).toSec();
        float dt = 1/IMU_HZ;
        E_(5, 0) = dt; 
        E_(6, 1) = dt;
        E_(7, 2) = dt;
        // x, y, psi, u, v, r, ax, ay

        // Predict step using IMU data        
        x_[0] = x_[0] + (x_[3]*cos(x_[2])-x_[4]*sin(x_[2]))*dt;
        x_[1] = x_[1] + (x_[3]*sin(x_[2])+x_[4]*cos(x_[2]))*dt;
        x_[2] = x_[2] + z_imu[1]*dt;
        x_[3] = x_[3] + z_imu[2]*dt + z_imu[1]*x_[4]*dt;
        x_[4] = x_[4] + z_imu[3]*dt - z_imu[1]*x_[3]*dt;
        x_[5] = x_[5];
        x_[6] = x_[6];
        x_[7] = x_[7];

        A_ << 1, 0, (-x_[3]*sin(x_[2])-x_[4]*cos(x_[2]))*dt, cos(x_[2])*dt, -sin(x_[2])*dt, 0, 0, 0,
              0, 1, (x_[3]*cos(x_[2])-x_[4]*sin(x_[2]))*dt, sin(x_[2])*dt, cos(x_[2])*dt, 0, 0, 0,
              0, 0, 1, 0, 0, dt, 0, 0,
              0, 0, 0, 1, 0, 0, dt, 0,
              0, 0, 0, 0, 1, 0, 0, dt,
              0, 0, 0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 0, 0, 1;

        P_ = A_ * P_ * A_.transpose() + E_ * Q_ * E_.transpose();

        
        last_time_ = msg->header.stamp;
        x_[2] = calpi2pi(x_[2]);
        publishState();
        }
    }

    void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
        // Update step using GPS data
        double easting, northing;
        int zone;
        bool northp;

        // Convert to UTM 
        LatLongToUTM(msg->latitude, msg->longitude, easting, northing, zone, northp); // Output: easting, northing;        
        z_gps = VectorXd(2);
        z_gps << easting, northing;

        // Correction code
        if (!start_EKF)
        {
            x_[0] = easting;
            x_[1] = northing;        
            start_EKF = true;
        }
        else
        {
            VectorXd z_measure(6);
            z_measure << z_gps[0], z_gps[1], z_imu[0], z_imu[1], z_imu[2], z_imu[3];
            
            // std::cout << "Here is the matrix m:\n" << K_ << std::endl;        
            K_ = P_ * C_.transpose() * (C_ * P_ * C_.transpose() + R_).inverse();
            IKC_ = Eigen::MatrixXd::Identity(nx,nx) - K_*C_;
            P_ = IKC_ * P_ * IKC_.transpose() + K_ * R_ * K_.transpose();
            VectorXd eps(ngps+nimu);
            eps = z_measure - C_ * x_;
            x_ = x_ + K_ * eps;


            // std::cout << "Here is the EKF result:\n" << x_ << std::endl;        

            publishState();
        }    
    }

    // Function to convert latitude and longitude to UTM coordinates
    void LatLongToUTM(double lat, double lon, double& easting, double& northing, int& zone, bool& northp) 
    {
        // Convert latitude and longitude to radians
        lat = lat * PI / 180.0;
        lon = lon * PI / 180.0;

        // Calculate the UTM zone
        zone = int((lon + PI) / (6 * PI / 180.0)) + 1;

        // Central meridian of the zone
        double lon0 = (zone - 1) * 6 - 180 + 3;
        lon0 = lon0 * PI / 180.0;

        // Calculate ellipsoid parameters
        double N = a / sqrt(1 - pow(sin(lat), 2) * (2 * f - pow(f, 2)));
        double T = pow(tan(lat), 2);
        double C = (2 * f - pow(f, 2)) * pow(cos(lat), 2) / (1 - pow(sin(lat), 2) * (2 * f - pow(f, 2)));
        double A = cos(lat) * (lon - lon0);

        // Calculate M
        double M = a * ((1 - (2 * f - pow(f, 2)) / 4 - 3 * pow((2 * f - pow(f, 2)), 2) / 64 - 5 * pow((2 * f - pow(f, 2)), 3) / 256) * lat
                    - ((3 * (2 * f - pow(f, 2))) / 8 + 3 * pow((2 * f - pow(f, 2)), 2) / 32 + 45 * pow((2 * f - pow(f, 2)), 3) / 1024) * sin(2 * lat)
                    + (15 * pow((2 * f - pow(f, 2)), 2) / 256 + 45 * pow((2 * f - pow(f, 2)), 3) / 1024) * sin(4 * lat)
                    - (35 * pow((2 * f - pow(f, 2)), 3) / 3072) * sin(6 * lat));

        // Calculate easting and northing
        easting = k0 * N * (A + (1 - T + C) * pow(A, 3) / 6 + (5 - 18 * T + pow(T, 2) + 72 * C - 58 * (2 * f - pow(f, 2))) * pow(A, 5) / 120) + 500000.0;
        northing = k0 * (M + N * tan(lat) * (pow(A, 2) / 2 + (5 - T + 9 * C + 4 * pow(C, 2)) * pow(A, 4) / 24 + (61 - 58 * T + pow(T, 2) + 600 * C - 330 * (2 * f - pow(f, 2))) * pow(A, 6) / 720));

        // Adjust northing for southern hemisphere
        northp = (lat >= 0);
        if (!northp) {
            northing += 10000000.0;
        }
    }


private:
    void publishState() {
        std_msgs::Float64MultiArray state_msg;
        state_msg.data.resize(12);
        state_msg.data[0] = x_(0);
        state_msg.data[1] = x_(1);
        state_msg.data[2] = x_(2);
        state_msg.data[3] = x_(3);
        state_msg.data[4] = x_(4);
        state_msg.data[5] = x_(5);
        state_msg.data[6] = z_gps[0];
        state_msg.data[7] = z_gps[1];
        state_msg.data[8] = z_imu[0];
        state_msg.data[9] = z_imu[1];
        state_msg.data[10] = z_imu[2];
        state_msg.data[11] = z_imu[3];
        state_pub_.publish(state_msg);                            
    }

    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_;
    ros::Subscriber imu_ang_sub_;
    ros::Subscriber imu_acc_sub_;
    
    ros::Subscriber gps_sub_;
    ros::Publisher state_pub_;

    bool start_EKF = false;

    VectorXd x_; // State vector [x, y, u, psi, a, r]
    MatrixXd P_; // State covariance matrix
    MatrixXd A_; // State transition matrix
    MatrixXd C_; // Measurement matrix
    MatrixXd E_; // Process noise matrix
    MatrixXd Q_; // Process noise covariance matrix [a, r]
    MatrixXd K_; // Kalman gain
    MatrixXd IKC_; // Kalman gain
    MatrixXd R_imu_; // Measurement noise covariance matrix for IMU
    MatrixXd R_gps_; // Measurement noise covariance matrix for GPS
    MatrixXd R_; // Final Measurement noise covariance
    VectorXd z_imu;
    VectorXd z_gps;
    VectorXd z_imu_linear_acc;
    VectorXd z_imu_ang_vel;
    ros::Time last_time_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_fusion_ekf");
    SensorFusionEKF ekf;
    ros::spin();
    return 0;
}
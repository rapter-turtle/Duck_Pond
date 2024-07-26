#include <ros/ros.h>
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
#define nx 6 
#define DEF_BIAS_accx 0.13

// x, y, psi, u, v, r, ax, ay

const double pi = 3.14159265358979323846;
const double a = 6378137.0; // WGS84 major axis
const double f = 1 / 298.257223563; // WGS84 flattening
const double k0 = 0.9996; // UTM scale factor
const float IMU_HZ = 15;

class SensorFusionEKF {
public:
    SensorFusionEKF() {
        imu_sub_ = nh_.subscribe("/wamv/sensors/imu/imu/data", 1, &SensorFusionEKF::imuCallback, this);
        gps_sub_ = nh_.subscribe("/wamv/sensors/gps/gps/fix", 1, &SensorFusionEKF::gpsCallback, this);
        state_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/ekf/estimated_state", 1);

        // Initialize the state and covariance matrices
        x_ = VectorXd(nx);
        P_ = MatrixXd(nx, nx);
        A_ = MatrixXd(nx, nx);
        C_ = MatrixXd(5, nx);
        K_ = MatrixXd(5, 5);
        IKC_ = MatrixXd(nx, nx);
        E_ = MatrixXd(nx, 2);
        Q_ = MatrixXd(2, 2);
        R_imu_ = MatrixXd(3, 3);
        R_gps_ = MatrixXd(2, 2);
        R_ = MatrixXd(5,5);

        x_ << 0, 0, 0, 0, 0, 0;
        P_ << 1, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0,
              0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 1;
        P_ = P_ * 1e1;
        C_ << 1, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 1;

        
        Q_ << 0.001, 0,
              0, 0.001;
        R_gps_ << 0.000001, 0,
                  0, 0.000001;
        R_imu_ << 0.00001, 0, 0,
                  0, 0.00001, 0,
                  0, 0, 0.0000001;
        R_.setZero();
        E_.setZero();
        R_.block<2,2>(0,0) = R_gps_;
        R_.block<3,3>(2,2) = R_imu_;
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
        z_imu = VectorXd(3);
        z_imu << yaw, msg->linear_acceleration.x-DEF_BIAS_accx, msg->angular_velocity.z;

        // double dt = (msg->header.stamp - last_time_).toSec();
        float dt = 1/IMU_HZ;
        E_(4, 0) = dt; E_(5, 1) = dt;

        // Predict step using IMU data        
        x_[0] = x_[0] + x_[2]*cos(x_[3])*dt;
        x_[1] = x_[1] + x_[2]*sin(x_[3])*dt;
        x_[2] = x_[2] + z_imu[1]*dt;
        x_[3] = x_[3] + z_imu[2]*dt;
        x_[4] = x_[4];
        x_[5] = x_[5];

        A_ << 1, 0, cos(x_[3])*dt, -x_[2]*sin(x_[3])*dt, 0, 0,
              0, 1, sin(x_[3])*dt, x_[2]*cos(x_[3])*dt, 0, 0,
              0, 0, 1, 0, dt, 0,
              0, 0, 0, 1, 0, dt,
              0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 1;

        P_ = A_ * P_ * A_.transpose() + E_ * Q_ * E_.transpose();

        
        last_time_ = msg->header.stamp;
        x_[3] = calpi2pi(x_[3]);
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
            VectorXd z_measure(5);
            z_measure << z_gps[0], z_gps[1], z_imu[0], z_imu[1], z_imu[2];
            
            // std::cout << "Here is the matrix m:\n" << K_ << std::endl;        
            K_ = P_ * C_.transpose() * (C_ * P_ * C_.transpose() + R_).inverse();
            IKC_ = Eigen::MatrixXd::Identity(6,6) - K_*C_;
            P_ = IKC_ * P_ * IKC_.transpose() + K_ * R_ * K_.transpose();
            VectorXd eps(5);
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
        state_msg.data.resize(10);
        state_msg.data[0] = x_(0);
        state_msg.data[1] = x_(1);
        state_msg.data[2] = x_(3);
        state_msg.data[3] = x_(2);
        state_msg.data[4] = x_(4);
        state_msg.data[5] = x_(5);
        state_msg.data[6] = z_gps[0];
        state_msg.data[7] = z_gps[1];
        state_msg.data[8] = z_imu[0];
        state_msg.data[9] = z_imu[2];
        state_pub_.publish(state_msg);                            
    }

    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_;
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

    ros::Time last_time_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_fusion_ekf_backup");
    SensorFusionEKF ekf;
    ros::spin();
    return 0;
}
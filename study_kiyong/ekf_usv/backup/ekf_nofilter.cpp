#include <ros/ros.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <Eigen/Dense>
#include <std_msgs/Float64MultiArray.h> //include the lib
#include <tf/tf.h>
#include <cmath>
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::Matrix2d;
using Eigen::Vector2d;

/* Parameters */
#define UTM_ZON 52
#define Wa 6378137
#define Wb 6356752.314245
#define We 0.081819190842965
#define Weps 0.006739496742333
#define PI 3.1415926535897931
#define DEF_DEG2RAD 3.1415926535897931 / 180.0
#define DEF_RAD2DEG 180.0 / 3.1415926535897931
#define DEF_BIAS_HDG 180*DEF_DEG2RAD
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
const float IMU_HZ = 100;
const float dt = 1/IMU_HZ;

class SensorFusionEKF {
public:
    SensorFusionEKF() {
        imu_sub_ = nh_.subscribe("/imu/data", 1, &SensorFusionEKF::imuCallback, this);
        gps_sub_ = nh_.subscribe("/ublox_gps/fix", 1, &SensorFusionEKF::gpsCallback, this);
        gps_vel_sub_ = nh_.subscribe("/ublox_gps/fix_velocity", 1, &SensorFusionEKF::gpsVelCallback, this);
        state_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/ekf/estimated_state", 1);
    
        z_imu = VectorXd(2);
        z_imu << 0.0, 0.0;
        z_gps_vel = VectorXd(2);
        z_gps_vel << 0.0, 0.0;
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
        tf::Quaternion q(
            msg->orientation.x,
            msg->orientation.y,
            msg->orientation.z,
            msg->orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw, yaw_rate;
        m.getRPY(roll, pitch, yaw);
        yaw = yaw + DEF_BIAS_HDG; // declination
        yaw = calpi2pi(yaw);
        yaw_rate = (yaw-z_imu[0])/dt;
        z_imu = VectorXd(2);
        z_imu << yaw, yaw_rate;
    }

    void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
        double easting, northing;
        int zone;
        bool northp;
        LatLongToUTM(msg->latitude, msg->longitude, easting, northing, zone, northp); // Output: easting, northing;        
        z_gps = VectorXd(2);
        z_gps << easting, northing;
        publishState();
    }

    void gpsVelCallback(const geometry_msgs::TwistWithCovarianceStamped::ConstPtr& msg) {
        Matrix2d A;
        A(0, 0) = cos(z_imu[0]); A(0, 1) =  sin(z_imu[0]);
        A(1, 0) = sin(z_imu[0]); A(1, 1) = -cos(z_imu[0]);
        Vector2d b;
        b(0) = msg->twist.twist.linear.x;
        b(1) = msg->twist.twist.linear.y;
        Vector2d u_v = A.colPivHouseholderQr().solve(b);
        // std::cout << "u_v: [" << u_v(0) << ", " << u_v(1) << "]" << std::endl;

        z_gps_vel = VectorXd(2);
        z_gps_vel << u_v(0), u_v(1); 
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
        // std::cout << "u_v: [" << z_gps(0) << ", " << z_gps(1) << "]" << std::endl;
        // std::cout << "u_v: [" << z_imu(0) << ", " << z_imu(1) << "]" << std::endl;
        // std::cout << "u_v: [" << z_gps_vel(0) << ", " << z_gps_vel(1) << "]" << std::endl;
        std_msgs::Float64MultiArray state_msg;
        state_msg.data.resize(6);
        state_msg.data[0] = z_gps[0];
        state_msg.data[1] = z_gps[1];
        state_msg.data[2] = z_imu[0];
        state_msg.data[3] = z_gps_vel[0];
        state_msg.data[4] = z_gps_vel[1];
        state_msg.data[5] = z_imu[1];
        state_pub_.publish(state_msg);                            
    }

    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_;    
    ros::Subscriber gps_sub_;
    ros::Subscriber gps_vel_sub_;
    ros::Publisher state_pub_;

    bool start_EKF = false;
    VectorXd z_imu;
    VectorXd z_gps;
    VectorXd z_gps_vel;
    ros::Time last_time_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_fusion_ekf");
    SensorFusionEKF ekf;
    ros::spin();
    return 0;
}
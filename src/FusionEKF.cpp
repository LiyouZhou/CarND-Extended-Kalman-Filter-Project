#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Q_v      = MatrixXd(2, 2);

    // measurement matrix - laser
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    // measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    // measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    // Set the process and measurement noises
    // Use noise_ax = 9 and noise_ay = 9 for the Q matrix.
    float noise_ax = 9;
    float noise_ay = 9;

    // construct the constant part of Q first.
    Q_v << noise_ax*noise_ax, 0,
           0, noise_ay*noise_ay;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        // first measurement
        VectorXd x_in(4);
        x_in << 0, 0, 0, 0;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert radar from polar to cartesian coordinates and initialize state.
            float ro = measurement_pack.raw_measurements_(0);
            float theta = measurement_pack.raw_measurements_(1);
            float ro_dot = measurement_pack.raw_measurements_(2);
            x_in(0) = ro * sin(theta);
            x_in(1) = ro * cos(theta);
            x_in(2) = ro_dot * sin(theta);
            x_in(3) = ro_dot * cos(theta);
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            // LASER only measures location leave velocity default.
            x_in(0) = measurement_pack.raw_measurements_(0);
            x_in(1) = measurement_pack.raw_measurements_(1);
        }

        previous_timestamp_ = measurement_pack.timestamp_;

        // Initialise matrices in ekf object

        // state x
        ekf_.x_ = x_in;

        // state transition matrix
        ekf_.F_ = MatrixXd(4, 4);
        ekf_.F_ << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

        //state covariance matrix P
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1000, 0,
                   0, 0, 0, 1000;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    // - Time is measured in seconds.
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    // Update the state transition matrix F according to the new elapsed time.
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // Update the process noise covariance matrix.
    MatrixXd G(4, 2);
    G << dt*dt/2.0, 0,
         0, dt*dt/2.0,
         dt, 0,
         0, dt;
    ekf_.Q_ = G * Q_v * G.transpose();

    // prediction step
    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    // Use the sensor type to perform the update step.
    // Update the state and covariance matrices.
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.R_ = R_radar_; // set the correct measurement covariance matrix
        // linearise the measurement functions around the current state point
        ekf_.H_ = Tools::CalculateJacobian(ekf_.x_);
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Laser updates
        ekf_.R_ = R_laser_; // set the correct measurement covariance matrix
        ekf_.H_ = H_laser_; // set the correct measurement matrix
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << endl << ekf_.x_ << endl;
    cout << "P_ = " << endl << ekf_.P_ << endl;
}

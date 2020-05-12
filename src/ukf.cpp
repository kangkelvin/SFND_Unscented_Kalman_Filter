#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.setIdentity();

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 15;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  // set initialised as false to wait for first measurement
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   *
   * This is the only function being called at the main function. Hence this
   * function will contain the
   * 1. predict state step (agnostic of sensor type)
   * 2. update measurement and state (lidar / radar)
   */

  // process first measurement
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(kState::px) = meas_package.raw_measurements_(kLidar::x);
      x_(kState::py) = meas_package.raw_measurements_(kLidar::y);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double r = meas_package.raw_measurements_(kRadar::r);
      double phi = meas_package.raw_measurements_(kRadar::phi);
      x_(kState::px) = r * cos(phi);
      x_(kState::py) = r * sin(phi);
    }
    x_(kState::v) = 0;
    x_(kState::yaw) = 0;
    x_(kState::yawd) = 0;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
}

void UKF::Prediction(double dt) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}

void UKF::phiGuard(double &phi) {
  while (phi > M_PI) phi -= M_PI;
  while (phi < -1 * M_PI) phi += M_PI;
}
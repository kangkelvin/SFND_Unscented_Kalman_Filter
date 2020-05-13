#include <iostream>

#include "Eigen/Dense"
#include "ukf.h"

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
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

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
  n_sig_ = 2 * n_aug_ + 1;
  lambda_ = 3 - n_aug_;
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);

  weights_ = VectorXd::Zero(n_sig_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  double weightNonZero = 1 / (2 * (lambda_ + n_aug_));
  for (int i = 1; i < weights_.size(); ++i) {
    weights_(i) = weightNonZero;
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage &meas_package) {
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
    switch (meas_package.sensor_type_) {
      case MeasurementPackage::LASER:
        x_(kState::px) = meas_package.raw_measurements_(kLidar::x);
        x_(kState::py) = meas_package.raw_measurements_(kLidar::y);
        break;
      case MeasurementPackage::RADAR:
        double r = meas_package.raw_measurements_(kRadar::r);
        double phi = meas_package.raw_measurements_(kRadar::phi);
        x_(kState::px) = r * cos(phi);
        x_(kState::py) = r * sin(phi);
        break;
    }
    x_(kState::v) = 0;
    x_(kState::yaw) = 0;
    x_(kState::yawd) = 0;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double dt = (double)(meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  this->Prediction(dt);

  switch (meas_package.sensor_type_) {
    case MeasurementPackage::LASER:
      this->UpdateLidar(meas_package);
      break;
    case MeasurementPackage::RADAR:
      this->UpdateRadar(meas_package);
      break;
  }
}

void UKF::Prediction(double dt) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

  /////////////////////////// Find Sigma Points ///////////////////////////
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();

  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    VectorXd secondTerm = sqrt(lambda_ + n_aug_) * P_aug_sqrt.col(i);
    Xsig_aug.col(i + 1) = x_aug + secondTerm;
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - secondTerm;
  }

  /////////////////////////// Predict Sigma Points ///////////////////////////
  for (int i = 0; i < n_sig_; ++i) {
    double px = Xsig_aug(kState::px, i);
    double py = Xsig_aug(kState::py, i);
    double v = Xsig_aug(kState::v, i);
    double yaw = Xsig_aug(kState::yaw, i);
    double yawd = Xsig_aug(kState::yawd, i);
    double nu_a = Xsig_aug(kState::nu_a, i);
    double nu_yawdd = Xsig_aug(kState::nu_yawdd, i);

    // incremental state change vector
    VectorXd delta = VectorXd::Zero(n_x_);

    // process noise vector
    VectorXd nu = VectorXd::Zero(n_x_);

    nu(kState::px) = 0.5 * dt * dt * cos(yaw) * nu_a;
    nu(kState::py) = 0.5 * dt * dt * sin(yaw) * nu_a;
    nu(kState::v) = dt * nu_a;
    nu(kState::yaw) = 0.5 * dt * dt * nu_yawdd;
    nu(kState::yawd) = dt * nu_yawdd;

    if (fabs(yawd) < 0.01) {
      delta(kState::px) = v * cos(yaw) * dt;
      delta(kState::py) = v * sin(yaw) * dt;
    } else {
      delta(kState::px) = v / yawd * (sin(yaw + yawd * dt) - sin(yaw));
      delta(kState::py) = v / yawd * (-cos(yaw + yawd * dt) + cos(yaw));
    }
    delta(kState::v) = 0;
    delta(kState::yaw) = yawd * dt;
    delta(kState::yawd) = 0;

    Xsig_pred_.col(i) = Xsig_aug.block(0, i, n_x_, 1) + delta + nu;
  }

  /////////////////////// Find Sigma mean and covariance /////////////////////
  x_pred_ = VectorXd::Zero(n_x_);
  P_pred_ = MatrixXd::Zero(n_x_, n_x_);

  for (int i = 0; i < n_sig_; ++i) {
    x_pred_ += weights_(i) * Xsig_pred_.col(i);
  }

  for (int i = 0; i < n_sig_; ++i) {
    VectorXd delta_x = Xsig_pred_.col(i) - x_pred_;
    phiGuard(delta_x(kState::yaw));
    P_pred_ += weights_(i) * delta_x * delta_x.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage &meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage &meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}

void UKF::phiGuard(double &phi) {
  while (phi > M_PI) phi -= 2.0 * M_PI;
  while (phi < -1 * M_PI) phi += 2.0 * M_PI;
}
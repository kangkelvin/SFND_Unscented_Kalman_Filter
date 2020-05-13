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
  P_ /= 10;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 10;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

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

  radarNisLimit = 7.815;
  lidarNisLimit = 5.991;
}

UKF::~UKF() {
  int overshootCount = 0;
  for (auto value : lidarNis) {
    if (value > lidarNisLimit) overshootCount++;
  }
  std::cout << "lidar NIS ratio: " << overshootCount * 1.0 / lidarNis.size()
            << std::endl;

  overshootCount = 0;
  for (auto value : radarNis) {
    if (value > radarNisLimit) overshootCount++;
  }
  std::cout << "radar NIS ratio: " << overshootCount * 1.0 / radarNis.size()
            << std::endl;
}

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
      case MeasurementPackage::SensorType::LASER:
        x_(kState::px) = meas_package.raw_measurements_(kLidar::x);
        x_(kState::py) = meas_package.raw_measurements_(kLidar::y);
        x_(kState::v) = 0;
        break;
      case MeasurementPackage::SensorType::RADAR:
        double r = meas_package.raw_measurements_(kRadar::r);
        double phi = meas_package.raw_measurements_(kRadar::phi);
        double rd = meas_package.raw_measurements_(kRadar::rd);
        x_(kState::px) = r * cos(phi);
        x_(kState::py) = r * sin(phi);
        x_(kState::v) = rd * cos(phi);
        break;
    }
    x_(kState::yaw) = 0;
    x_(kState::yawd) = 0;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  switch (meas_package.sensor_type_) {
    case MeasurementPackage::SensorType::LASER:
      if (use_laser_) {
        double dt = (double)(meas_package.timestamp_ - time_us_) / 1000000.0;
        time_us_ = meas_package.timestamp_;
        this->Prediction(dt);
        this->UpdateLidar(meas_package);
      }
      break;
    case MeasurementPackage::SensorType::RADAR:
      if (use_radar_) {
        double dt = (double)(meas_package.timestamp_ - time_us_) / 1000000.0;
        time_us_ = meas_package.timestamp_;
        this->Prediction(dt);
        this->UpdateRadar(meas_package);
      }
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

  // MatrixPhiGuard(Xsig_aug, kState::yaw);

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

  // MatrixPhiGuard(Xsig_pred_, kState::yaw);

  /////////////////////// Find Sigma mean and covariance /////////////////////
  x_.setZero();
  P_.setZero();

  for (int i = 0; i < n_sig_; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  std::cout << "X_pred:\n" << x_ << "\n\n";

  // MatrixPhiGuard(x_, kState::yaw);

  for (int i = 0; i < n_sig_; ++i) {
    VectorXd delta_x = Xsig_pred_.col(i) - x_;
    phiGuard(delta_x(kState::yaw));
    P_ += weights_(i) * delta_x * delta_x.transpose();
  }
  std::cout << "P_pred:\n" << P_ << "\n\n";
}

void UKF::UpdateLidar(MeasurementPackage &meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = meas_package.raw_measurements_.size();

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, n_sig_);

  // mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < n_sig_; ++i) {
    double px = Xsig_pred_(kState::px, i);
    double py = Xsig_pred_(kState::py, i);

    Zsig(kLidar::x, i) = px;
    Zsig(kLidar::y, i) = py;
  }

  // calculate mean predicted measurement
  for (int i = 0; i < n_sig_; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  for (int i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;

  S += R;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z = meas_package.raw_measurements_;

  // update state mean and covariance matrix
  x_ += K * (z - z_pred);
  P_ -= K * S * K.transpose();

  std::cout << "z:\n" << z << "\n\n";
  std::cout << "z_pred:\n" << z_pred << "\n\n";
  std::cout << "updated x_:\n" << x_ << "\n\n";

  lidarNis.emplace_back(calcNormInnovSquare(z, z_pred, S));
}

void UKF::UpdateRadar(MeasurementPackage &meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  int n_z = meas_package.raw_measurements_.size();

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, n_sig_);

  // mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < n_sig_; ++i) {
    double px = Xsig_pred_(kState::px, i);
    double py = Xsig_pred_(kState::py, i);
    double v = Xsig_pred_(kState::v, i);
    double yaw = Xsig_pred_(kState::yaw, i);
    double yawd = Xsig_pred_(kState::yawd, i);

    Zsig(kRadar::r, i) = sqrt(px * px + py * py);
    Zsig(kRadar::phi, i) = atan2(py, px);
    Zsig(kRadar::rd, i) =
        (px * cos(yaw) * v + py * sin(yaw) * v) / Zsig(kRadar::r, i);
  }

  // calculate mean predicted measurement
  for (int i = 0; i < n_sig_; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  MatrixPhiGuard(z_pred, kRadar::phi);

  // calculate innovation covariance matrix S
  for (int i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    MatrixPhiGuard(z_diff, kRadar::phi);
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;

  S += R;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    MatrixPhiGuard(z_diff, kRadar::phi);
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    MatrixPhiGuard(x_diff, kState::yaw);
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z = meas_package.raw_measurements_;

  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  MatrixPhiGuard(z_diff, kRadar::phi);
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  // std::cout << "K:\n" << K << "\n\n";
  std::cout << "z:\n" << z << "\n\n";
  std::cout << "z_pred:\n" << z_pred << "\n\n";
  std::cout << "updated x_:\n" << x_ << "\n\n";

  radarNis.emplace_back(calcNormInnovSquare(z, z_pred, S));
}

void UKF::phiGuard(double &phi) {
  while (phi > M_PI) {
    phi -= 2.0 * M_PI;
  }

  while (phi < -1 * M_PI) {
    phi += 2.0 * M_PI;
  }
}

void UKF::MatrixPhiGuard(MatrixXd &mtx, int rowPos) {
  for (int i = 0; i < mtx.cols(); ++i) {
    phiGuard(mtx(rowPos, i));
  }
}

void UKF::MatrixPhiGuard(VectorXd &vec, int rowPos) { phiGuard(vec(rowPos)); }

double UKF::calcNormInnovSquare(VectorXd &z, VectorXd &z_pred, MatrixXd &S) {
  VectorXd z_diff = z - z_pred;
  double e = z_diff.transpose() * S.inverse() * z_diff;
  return e;
}
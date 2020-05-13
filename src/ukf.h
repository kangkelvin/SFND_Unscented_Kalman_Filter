#ifndef UKF_H
#define UKF_H

#include <cmath>

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage &meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param dt Time between k and k+1 in s
   */
  void Prediction(double dt);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage &meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage &meas_package);

  /**
   * prevents yaw angle from going beyong -pi to pi
   * @param phi is the yaw angle that you want to restrict
   */
  void phiGuard(double &phi);

  /**
   * prevents yaw angle from going beyong -pi to pi
   * @param rowPos is the index of angle to be guarded
   */
  void MatrixPhiGuard(Eigen::MatrixXd &mtx, int rowPos);
  void MatrixPhiGuard(Eigen::VectorXd &vec, int rowPos);
  int smallYawd = 0;
  int normalYawd = 0;

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state vector after prediction step
  Eigen::VectorXd x_pred_;

  // state covariance matrix after prediction step
  Eigen::MatrixXd P_pred_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma points dimension
  int n_sig_;

  // Sigma point spreading parameter
  double lambda_;

  // enum for state matrix of vehicle
  enum kState { px, py, v, yaw, yawd, nu_a, nu_yawdd };

  // enum for lidar measurement points
  enum kLidar { x, y };

  // enum for radar measurement points
  enum kRadar { r, phi, rd };
};

#endif  // UKF_H
#include "kalman_filter.h"
#include <iostream>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

// update the state by using Kalman Filter equations
void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y  = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S  = H_ * P_ * Ht + R_;
  MatrixXd K  = P_ * Ht * S.inverse();

  // new state
  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;
}

// the non-linear radar measurement functions
static VectorXd h(const VectorXd& x_state)
{
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  float a = sqrt(px*px + py*py);

  VectorXd retval(3);
  retval << a,
            atan2f(py, px),
            (px*vx+py*vy)/a;

  return retval;
}

// update the state by using Extended Kalman Filter equations
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd y = z - h(x_);
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  // Clean up y(1) i.e. theta to be in range -pi +pi
  while (y(1) < -M_PI) y(1) += 2*M_PI;
  while (y(1) >  M_PI) y(1) -= 2*M_PI;

  // new state
  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;
}

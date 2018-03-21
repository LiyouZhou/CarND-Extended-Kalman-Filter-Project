#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if((estimations.size() != ground_truth.size()) ||
	   (estimations.size() == 0)) {
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i) {
        // ... your code here
        VectorXd diff(4);
        diff = estimations[i] - ground_truth[i];
        diff = diff.array() * diff.array();
        rmse += diff;
	}

	//calculate the mean
	rmse /= estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3,4);

	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float a = px*px + py*py;

	//check division by zero
	if(fabs(a) < 0.0001) {
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	float b = sqrt(a);
	float c = px/b;
	float d = py/b;
	float e = (vx*py-vy*px)/pow(b, 3.0);

	//compute the Jacobian matrix
	Hj << c, d, 0, 0,
	      -py/a, px/a, 0, 0,
	      py*e, px*e, c, d;

	return Hj;
}

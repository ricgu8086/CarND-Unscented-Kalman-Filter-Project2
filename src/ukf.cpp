#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 30;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 30;

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
	 TODO:

	 Complete the initialization. See ukf.h for other member properties.

	 Hint: one or more values initialized above might be wildly off...
	 */

	// time when the state is true, in us
	time_us_ = 0;

	// initially set to false, set to true in first call of ProcessMeasurement
	is_initialized_ = false;

	// State dimension
	n_x_ = 5;

	// Augmented state dimension
	n_aug_ = n_x_ + 2;

	// Number of Sigma points
	n_sig_ = 2*n_aug_+1;

	// Sigma point spreading parameter
	lambda_ = 3 - n_x_;

	// Weights of sigma points
	weights_ = VectorXd(n_aug_);

	weights_(0) = lambda_/(lambda_ + n_aug_);

	double w = 1/(2*(lambda_ + n_aug_));

	for(int i=1; i<n_sig_; i++)
		weights_(i) = w;

	// the current NIS for radar
	NIS_radar_ = 0;

	// the current NIS for laser
	NIS_laser_ = 0;
}

UKF::~UKF()
{
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
	/**
	 TODO:

	 Complete this function! Make sure you switch between lidar and radar
	 measurements.
	 */

	/* Prediction */
	/**************/

	double dt = meas_package.timestamp_; // TODO subtract previous timestamp
	Prediction(dt);



	/* Update */
	/*********/

	/* Predict Measurement */

	/* Update State */

	if(meas_package.sensor_type_ == MeasurementPackage::LASER)
		UpdateLidar(meas_package);
	else
		UpdateRadar(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
	/**
	 Complete this function! Estimate the object's location. Modify the state
	 vector, x_. Predict sigma points, the state, and the state covariance matrix.
	 */

	/* Generate Sigma Points */
	MatrixXd Sigma_p = GenerateSigmaPoints();

	/* Predict Sigma Points */
	MatrixXd Sigma_p_pred = PredictSigmaPoints(Sigma_p, delta_t);

	/* Predict Mean and Covariance */
	PredictMeanCovariance(Sigma_p_pred);
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
	/**
	 TODO:

	 Complete this function! Use lidar data to update the belief about the object's
	 position. Modify the state vector, x_, and covariance, P_.

	 You'll also need to calculate the lidar NIS.
	 */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
	/**
	 TODO:

	 Complete this function! Use radar data to update the belief about the object's
	 position. Modify the state vector, x_, and covariance, P_.

	 You'll also need to calculate the radar NIS.
	 */
}

/**
 * Normalize an angle between -M_PI and M_PI
 */
double UKF::NormalizeAngle(double angle)
{
	if (angle > M_PI)
	{
		double temp = fmod((angle - M_PI), (2 * M_PI)); // -= 2. * M_PI;
		angle = temp - M_PI;
	}
	if (angle < -M_PI)
	{
		double temp = fmod((angle + M_PI), (2 * M_PI));
		angle = temp + M_PI;
	}
	return angle;
}

/**
 * Apply the CTRV process model
 */
VectorXd UKF::ProcessModel(VectorXd SigmaPoint, double delta_t)
{
	VectorXd SigmaPointPred(n_x_), x, operand1(n_x_), operand2(n_x_);
	double v, psi, psi_dot, nu_a, nu_psi_dotdot;


	x = SigmaPoint;

	v = x(2);
	psi = x(3);
	psi_dot = x(4);
	nu_a = x(5);
	nu_psi_dotdot = x(6);

	if (fabs(psi_dot) < 0.0001 ) // it's zero
	{
		operand1 << v*cos(psi)*delta_t,
					v*sin(psi)*delta_t,
					0,
					psi_dot*delta_t,
					0;
	}
	else //it's not zero
	{
		operand1 << max(v/psi_dot, 0.0001)*(sin(psi + psi_dot*delta_t) - sin(psi)),
					max(v/psi_dot, 0.0001)*(-cos(psi + psi_dot*delta_t) + cos(psi)),
					0,
					psi_dot*delta_t,
					0;
	}

	operand2 << 1/2.0*delta_t*delta_t*cos(psi)*nu_a,
				1/2.0*delta_t*delta_t*sin(psi)*nu_a,
				delta_t*nu_a,
				1/2.0*delta_t*delta_t*nu_psi_dotdot,
				delta_t*nu_psi_dotdot;

	SigmaPointPred = x.head(5) + operand1 + operand2;

	return SigmaPointPred;
}

/**
 * Generate the sigma points that represent the distribution
 */
MatrixXd UKF::GenerateSigmaPoints(void)
{
	MatrixXd sigma_p(n_aug_, n_sig_);

	sigma_p.col(0) = x_;

	//calculate square root of P
	MatrixXd A = P_.llt().matrixL();

	double scaling_factor = sqrt(lambda_ + n_aug_);

	for(int i=1; i<n_aug_; i++)
	{
		sigma_p.col(i) = x_ + scaling_factor*A.col(i);
		sigma_p.col(i + n_aug_) = x_ - scaling_factor*A.col(i);
	}

	return sigma_p;
}

/**
 * Move the sigma points through the process model
 */
MatrixXd UKF::PredictSigmaPoints(MatrixXd Sigma_p, double delta_t)
{
	MatrixXd predicted(n_aug_, n_sig_);

	for(int i=0; i<n_sig_; i++)
		predicted.col(i) = ProcessModel(Sigma_p.col(i), delta_t);

	return predicted;
}

/**
 * Recover the approximate gaussian distribution from predicted sigma points
 */
void UKF::PredictMeanCovariance(MatrixXd Sigma_p_pred)
{
	// Mean
	x_.setZero();

	for(int i=0; i<n_sig_; i++)
		x_ += weights_(i)*Sigma_p_pred.col(i);

	// Covariance
	P_.setZero();

	VectorXd x_diff;

	for(int i=0; i<n_sig_; i++)
	{
		x_diff = Sigma_p_pred.col(i)- x_;
		P_ += weights_(i)*x_diff*x_diff.transpose();
	}
}

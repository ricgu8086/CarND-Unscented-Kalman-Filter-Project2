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
	x_.setZero();

	// initial covariance matrix
	P_ = MatrixXd(5, 5);
	P_.setZero();

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 2.0;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.3;

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
	 Complete the initialization. See ukf.h for other member properties.
	 Hint: one or more values initialized above might be wildly off...
	 */

	// initially set to false, set to true in first call of ProcessMeasurement
	is_initialized_ = false;

	// time when the state is true, in us
	previous_timestamp_ = 0.0;

	// Previous timestamp in microseconds
	previous_timestamp_ = 0.0;

	// State dimension
	n_x_ = 5;

	// Augmented state dimension
	n_aug_ = n_x_ + 2;

	// Number of Sigma points
	n_sig_ = 2*n_aug_+1;

	// Sigma point spreading parameter
	lambda_ = 3 - n_aug_;

	// predicted sigma points matrix
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// Weights of sigma points
	weights_ = VectorXd(2 * n_aug_ + 1);

	weights_(0) = lambda_ / (lambda_ + n_aug_);

	double w = 1/(2*(lambda_ + n_aug_));

	for(int i=1; i<2 * n_aug_ + 1; i++)
		weights_(i) = w;

	// Measurement noise for radar
	R_radar_ = MatrixXd(3, 3);

	R_radar_ << std_radr_*std_radr_ , 0							, 0,
				0					, std_radphi_*std_radphi_	, 0,
				0					, 0							, std_radrd_*std_radrd_;

	// Measurement noise for lidar
	R_lidar_ = MatrixXd(2, 2);

	R_lidar_ << std_laspx_*std_laspx_	, 0,
				0						, std_laspy_*std_laspy_;

	// the current NIS for radar
	NIS_radar_ = 0.0;

	// the current NIS for laser
	NIS_laser_ = 0.0;
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
	 Complete this function! Make sure you switch between lidar and radar
	 measurements.
	 */

	// skip predict/update if sensor type is ignored
	if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
		|| (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_))
	{
		if (!is_initialized_)
		{
			//Initialize state x, P and timestamp

			// first measurement
			x_ << 1, 1, 1, 1, 0.1;

			// init covariance matrix
			P_ << 	0.15, 0		, 0		, 0		, 0,
					0	, 0.15	, 0		, 0		, 0,
					0	, 0		, 1		, 0		, 0,
					0	, 0		, 0		, 1		, 0,
					0	, 0		, 0		, 0		, 1;

			if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
			{
				x_(0) = meas_package.raw_measurements_(0);
				x_(1) = meas_package.raw_measurements_(1);
			}
			else if (meas_package.sensor_type_ == MeasurementPackage::RADAR	&& use_radar_)
			{
				// Convert radar from polar to cartesian coordinates and initialize state.
				double ro = meas_package.raw_measurements_(0);
				double phi = meas_package.raw_measurements_(1);

				x_(0) = ro * cos(phi);
				x_(1) = ro * sin(phi);
			}

			// For safety reasons. This code avoides arithmetic problems if the object is at the location of the measuring instrument.
			if ( fabs(x_(0)) < 0.001 && fabs(x_(1)) < 0.0001 )  
			{
			   x_(0) = 0.001;
			   x_(0) = 0.001;
			}

			previous_timestamp_ = meas_package.timestamp_;
			is_initialized_ = true;

			return;
		}

		/* Prediction */
		/**************/

		double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0; // dt in seconds
		previous_timestamp_ = meas_package.timestamp_;

		// The following block helps to avoid numerical instability during prediction when dt is too big (suggested by the reviewer).
		double max_step = 0.1;

		while (dt > 0.2) 
		{
		    Prediction(max_step);
		    dt -= max_step;
		}

		Prediction(dt);

		/* Update */
		/*********/

		if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			UpdateLidar(meas_package);
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			UpdateRadar(meas_package);
		}
	}
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

	/* Generate the sigma points that represent the distribution */

	//create sigma point matrix
	MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

	//calculate square root of P
	MatrixXd A = P_.llt().matrixL();

	//set lambda for non-augmented sigma points
	lambda_ = 3 - n_x_;

	//set first column of sigma point matrix
	Xsig.col(0) = x_;

	//set remaining sigma points
	for (int i = 0; i < n_x_; i++)
	{
		Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
		Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
	}

	// Augmentation step

	// Create augmented mean state x
	VectorXd x_augmented(n_aug_);
	x_augmented.setZero();
	x_augmented.head(5) = x_;

	// Create augmented covariance matrix p
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.setZero();
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);

	//set lambda for augmented sigma points
	lambda_ = 3 - n_aug_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_augmented;

	double scaling_factor = sqrt(lambda_ + n_aug_);

	for (int i=0; i<n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_augmented + scaling_factor * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_augmented - scaling_factor * L.col(i);
	}

	/*  Predict Sigma Points */
	/*************************/

	double px, py, v, yaw, yaw_dot, nu_a, nu_yaw_dot, px_p, py_p, v_p, yaw_p, yaw_dot_p;

	for (int i=0; i<n_sig_; i++)
	{
		px = Xsig_aug(0, i);
		py = Xsig_aug(1, i);
		v = Xsig_aug(2, i);
		yaw = Xsig_aug(3, i);
		yaw_dot = Xsig_aug(4, i);
		nu_a = Xsig_aug(5, i);
		nu_yaw_dot = Xsig_aug(6, i);


		// avoid division by zero
		if (fabs(yaw_dot > 0.001))
		{
			px_p = px + v / yaw_dot * (sin(yaw + yaw_dot * delta_t) - sin(yaw));
			py_p = py + v / yaw_dot * (cos(yaw) - cos(yaw + yaw_dot * delta_t));
		}
		else  // it's zero
		{
			px_p = px + v * delta_t * cos(yaw);
			py_p = py + v * delta_t * sin(yaw);
		}

		v_p = v;
		yaw_p = yaw + yaw_dot * delta_t;
		yaw_dot_p = yaw_dot;

		// add noise
		px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
		py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
		v_p = v_p + nu_a * delta_t;

		yaw_p = yaw_p + 0.5 * nu_yaw_dot * delta_t * delta_t;
		yaw_dot_p = yaw_dot_p + nu_yaw_dot * delta_t;

		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yaw_dot_p;
	}

	/* Recover the approximate gaussian distribution from predicted sigma points */

	x_.setZero();

	for (int i=0; i <n_sig_; i++)
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);

	P_.setZero();

	VectorXd x_diff;

	for (int i=0; i<n_sig_; i++)
	{
		x_diff = Xsig_pred_.col(i) - x_;
		x_diff(3) = NormalizeAngle(x_diff(3));

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
	/**
	 Complete this function! Use lidar data to update the belief about the object's
	 position. Modify the state vector, x_, and covariance, P_.
	 You'll also need to calculate the lidar NIS.
	 */

	VectorXd z = meas_package.raw_measurements_;
	int n_z = 2;

	MatrixXd Zsig = MatrixXd(n_z, n_sig_);

	for (int i=0; i<n_sig_; i++)
	{
		Zsig(0, i) = Xsig_pred_(0, i); // px
		Zsig(1, i) = Xsig_pred_(1, i); // py
	}

	//mean predicted measurement
	VectorXd z_pred(n_z);
	z_pred.setZero();

	for (int i=0; i<n_sig_; i++)
		z_pred = z_pred + weights_(i) * Zsig.col(i);

	//measurement covariance matrix S
	MatrixXd S(n_z, n_z);
	S.setZero();

	VectorXd z_diff;

	for (int i=0; i<n_sig_; i++)
	{
		z_diff = Zsig.col(i) - z_pred;
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	S = S + R_lidar_;


	/*  UKF Update for Lidar*/
	/************************/

	//cross correlation matrix Tc
	MatrixXd Tc(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.setZero();
	VectorXd x_diff;

	for (int i=0; i<n_sig_; i++)
	{
		z_diff = Zsig.col(i) - z_pred;
		x_diff = Xsig_pred_.col(i) - x_;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd S_inv = S.inverse();
	MatrixXd K = Tc * S_inv;

	z_diff = z - z_pred;

	//calculate NIS
	NIS_laser_ = z_diff.transpose() * S_inv * z_diff;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
	/**
	 Complete this function! Use radar data to update the belief about the object's
	 position. Modify the state vector, x_, and covariance, P_.

	 You'll also need to calculate the radar NIS.
	 */

	VectorXd z = meas_package.raw_measurements_;

	int n_z = 3;


	MatrixXd Zsig = MatrixXd(n_z, n_sig_);

	double px, py, v, yaw, v1, v2;

	for (int i=0; i<n_sig_; i++)
	{
		px = Xsig_pred_(0, i);
		py = Xsig_pred_(1, i);
		v = Xsig_pred_(2, i);
		yaw = Xsig_pred_(3, i);

		v1 = cos(yaw) * v;
		v2 = sin(yaw) * v;

		// measurement model
		px = px < 0.001 ? 0.001 : px; // This is done to avoid arithmetic problems, i.e. division by zero
		py = py < 0.001 ? 0.001 : py;
		Zsig(0, i) = sqrt(px*px + py*py);                       // r
		Zsig(1, i) = atan2(py, px);                   			// phi
		Zsig(2, i) = (px*v1 + py*v2) / sqrt(px*px + py*py);   	// r_dot
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.setZero();

	for (int i=0; i<n_sig_; i++)
		z_pred = z_pred + weights_(i) * Zsig.col(i);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.setZero();

	VectorXd z_diff;

	for (int i=0; i<n_sig_; i++)
	{
		z_diff = Zsig.col(i) - z_pred;
		z_diff(1) = NormalizeAngle(z_diff(1));

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	S = S + R_radar_;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	/*****************************************************************************
	 *  UKF Update for Radar
	 ****************************************************************************/


	//calculate cross correlation matrix
	Tc.setZero();

	VectorXd x_diff;

	for (int i=0; i<n_sig_; i++)
	{
		z_diff = Zsig.col(i) - z_pred;
		z_diff(1) = NormalizeAngle(z_diff(1));

		x_diff = Xsig_pred_.col(i) - x_;
		x_diff(3) = NormalizeAngle(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd S_inv = S.inverse();
	MatrixXd K = Tc * S_inv;

	z_diff = z - z_pred;
	z_diff(1) = NormalizeAngle(z_diff(1));

	//calculate NIS
	NIS_radar_ = z_diff.transpose() * S_inv * z_diff;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

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


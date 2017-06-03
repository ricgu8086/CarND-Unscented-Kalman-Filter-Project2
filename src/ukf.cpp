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
	std_a_ = 0.355;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 1.35;

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

	// Previous timestamp in microseconds
	previous_timestamp_ = 0;

	// initially set to false, set to true in first call of ProcessMeasurement
	is_initialized_ = false;

	// State dimension
	n_x_ = 5;

	// Augmented state dimension
	n_aug_ = n_x_ + 2;

	// Number of Sigma points
	n_sig_ = 2*n_aug_+1;

	// Dimension of Lidar measurement space
	n_z_lidar_ = 2;

	// Dimension of Radar measurement space
	n_z_radar_ = 3;

	// Measurement noise for radar
	R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);

	R_radar_ << std_radr_*std_radr_ , 0							, 0,
				0					, std_radphi_*std_radphi_	, 0,
				0					, 0							, std_radrd_*std_radrd_;

	// Measurement noise for lidar
	R_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);

	R_lidar_ << std_laspx_*std_laspx_	, 0,
				0						, std_laspy_*std_laspy_;

	// Sigma point spreading parameter
	lambda_ = 3 - n_x_;

	// Weights of sigma points
	weights_ = VectorXd(n_sig_);

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

	 Complete this function! Make sure you switch between lidar and radar
	 measurements.
	 */

	if(!is_initialized_)
	{
		//Initialize state x, P and timestamp
		if(meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
		}
		else
		{
			// Convert radar from polar to cartesian coordinates and initialize state.

			double rho = meas_package.raw_measurements_[0];
			double phi = meas_package.raw_measurements_[1];
			x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
		}

		P_ << 	0.02	, 0		, 0		, 0		, 0		,
				0		, 0.02	, 0		, 0		, 0		,
				0		, 0		, 0.2	, 0		, 0		,
				0		, 0		, 0		, 0.1	, 0		,
				0		, 0		, 0		, 0		, 0.1	;

		previous_timestamp_ = meas_package.timestamp_;
		is_initialized_ = true;

		return;

	}

	/* Prediction */
	/**************/

	double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0; // dt in seconds

	MatrixXd x_sigma_pred = Prediction(dt);



	/* Update */
	/*********/

	Update(x_sigma_pred, meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
MatrixXd UKF::Prediction(double delta_t)
{
	/**
	 Complete this function! Estimate the object's location. Modify the state
	 vector, x_. Predict sigma points, the state, and the state covariance matrix.
	 */

	/* Generate Sigma Points */
	MatrixXd x_sigma = GenerateSigmaPoints();

	/* Predict Sigma Points */
	MatrixXd x_sigma_pred = PredictSigmaPoints(x_sigma, delta_t);

	/* Predict Mean and Covariance */
	PredictMeanCovariance(x_sigma_pred);

	return x_sigma_pred;
}

/**
 * Generate the sigma points that represent the distribution
 */
MatrixXd UKF::GenerateSigmaPoints(void)
{
	// Augmentation step

	// Create augmented mean state x
	VectorXd x_augmented(n_aug_);
	x_augmented.setZero();
	x_augmented.head(n_x_) = x_;

	// Create augmented covariance matrix p
	MatrixXd p_augmented(n_aug_, n_aug_);
	p_augmented.setZero();
	p_augmented.topLeftCorner(n_x_, n_x_) = P_;
	p_augmented(5,5) = std_a_*std_a_;
	p_augmented(6,6) = std_yawdd_*std_yawdd_;



	MatrixXd sigma_p(n_aug_, n_sig_);


	sigma_p.col(0) = x_augmented;

	//calculate square root of P
	MatrixXd A = p_augmented.llt().matrixL();

	double scaling_factor = sqrt(lambda_ + n_aug_);

	for(int i=0; i<n_aug_; i++) // for my future self: n_aug_ is correct, don't use n_sig_
	{
		sigma_p.col(i + 1) = x_augmented + scaling_factor*A.col(i);
		sigma_p.col(i + 1 + n_aug_) = x_augmented - scaling_factor*A.col(i);
	}

	return sigma_p;
}

/**
 * Move the sigma points through the process model
 */
MatrixXd UKF::PredictSigmaPoints(MatrixXd Sigma_p, double delta_t)
{
	MatrixXd predicted(n_x_, n_sig_);

	for(int i=0; i<n_sig_; i++)
		predicted.col(i) = ProcessModel(Sigma_p.col(i), delta_t);

	return predicted;
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

	SigmaPointPred = x.head(n_x_) + operand1 + operand2;

	return SigmaPointPred;
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

	x_(3) = NormalizeAngle(x_(3));

	// Covariance
	P_.setZero();

	VectorXd x_diff;

	for(int i=0; i<n_sig_; i++)
	{
		x_diff = Sigma_p_pred.col(i) - x_;
		x_diff(3) = NormalizeAngle(x_diff(3));
		P_ += weights_(i)*x_diff*x_diff.transpose();
	}
}

void UKF::Update(MatrixXd x_sigma_pred, MeasurementPackage meas_package)
{
	pair< pair<VectorXd, MatrixXd>, MatrixXd> predicted;

	/* Predict Measurement */
	if(meas_package.sensor_type_ == MeasurementPackage::LASER)
	{
		predicted = PredictMeasurementLidar(x_sigma_pred);
	}
	else
	{
		predicted = PredictMeasurementRadar(x_sigma_pred);
	}

	/* Update State */
	UpdateState(predicted, x_sigma_pred, meas_package);

}

/**
 * Convert the sigma points from the state space to the Radar measurement space.
 */
pair< pair<VectorXd, MatrixXd>, MatrixXd>  UKF::PredictMeasurementRadar(MatrixXd sigma_points)
{
	// create matrix for sigma points in measurement space
	MatrixXd z_sig = MatrixXd(n_z_radar_, n_sig_);

	// transform sigma points into measurement space
	struct
	{
		VectorXd transform(VectorXd sigma_p, int n_z_radar_)
		{
			double px = sigma_p(0);
			double py = sigma_p(1);
			double v = sigma_p(2);
			double psi = sigma_p(3);
			double psi_dot = sigma_p(4);

			double rho = sqrt(px*px + py*py); // rho
			double phi = atan2(py, px); // phi
			double rho_dot = (px*cos(psi)*v + py*sin(psi)*v)/max(rho, 0.0001); // avoiding division by 0

			VectorXd measurement(n_z_radar_);
			measurement << rho, phi, rho_dot;

			return measurement;
		}
	} h_radar;

	for(int i=0; i<n_sig_; i++)
		z_sig.col(i) = h_radar.transform(sigma_points.col(i), n_z_radar_);


	//calculate mean predicted measurement
	VectorXd z_pred = VectorXd(n_z_radar_);

	z_pred.setZero();

	for(int i=0; i<n_sig_; i++)
		z_pred += weights_(i)*z_sig.col(i);

	//calculate measurement covariance matrix
	MatrixXd S = MatrixXd(n_z_radar_,n_z_radar_);

	S.setZero();
	VectorXd factor(n_z_radar_);

	for(int i=0; i<n_sig_; i++)
	{
	  factor = z_sig.col(i) - z_pred;
	  //Normalization
	  factor(1) = NormalizeAngle(factor(1));
	  S += weights_(i)*factor*factor.transpose();
	}

	//add measurement noise covariance matrix
	S += R_radar_;

	return make_pair(make_pair(z_pred, z_sig), S);
}

/**
 * Convert the sigma points from the state space to the Lidar measurement space.
 */
pair< pair<VectorXd, MatrixXd>, MatrixXd> UKF::PredictMeasurementLidar(MatrixXd x_sigma_pred)
{
	// create matrix for sigma points in measurement space
	MatrixXd z_sig = MatrixXd(n_z_lidar_, n_sig_);

	// transform sigma points into measurement space
	struct
	{
		VectorXd transform(VectorXd sigma_p, int n_z_lidar_)
		{
			double px = sigma_p(0);
			double py = sigma_p(1);
			double v = sigma_p(2);
			double psi = sigma_p(3);
			double psi_dot = sigma_p(4);

			VectorXd measurement(n_z_lidar_);
			measurement << px, py;
			return measurement;
		}
	} h_lidar;

	for(int i=0; i<n_sig_; i++)
		z_sig.col(i) = h_lidar.transform(x_sigma_pred.col(i), n_z_lidar_);


	//calculate mean predicted measurement
	VectorXd z_pred = VectorXd(n_z_lidar_);

	z_pred.setZero();

	for(int i=0; i<n_sig_; i++)
		z_pred += weights_(i)*z_sig.col(i);

	//calculate measurement covariance matrix
	MatrixXd S = MatrixXd(n_z_lidar_,n_z_lidar_);

	S.setZero();
	VectorXd factor(n_z_lidar_);

	for(int i=0; i<n_sig_; i++)
	{
	  factor = z_sig.col(i) - z_pred;
	  //Normalization
	  factor(1) = NormalizeAngle(factor(1));
	  S += weights_(i)*factor*factor.transpose();
	}

	//add measurement noise covariance matrix
	S += R_lidar_;

	return make_pair(make_pair(z_pred, z_sig), S);
}

/**
 * Updates the state and the state covariance matrix using a measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateState(pair< pair<VectorXd, MatrixXd>, MatrixXd> predicted, MatrixXd x_sigma_pred, MeasurementPackage meas_package)
{
	/**
	 Complete this function! Use radar data to update the belief about the object's
	 position. Modify the state vector, x_, and covariance, P_.

	 You'll also need to calculate the radar NIS.
	 */

	// Unpack variables (really need c++11 tuples for this)
	pair<VectorXd, MatrixXd> z_estimated = predicted.first;
	VectorXd z_pred = z_estimated.first;
	MatrixXd z_sig = z_estimated.second;
	MatrixXd S = predicted.second;

	// Configure if it's a Radar or a Lidar measurement
	int n_measurement;
	double *NIS_p;

	if(meas_package.sensor_type_ == MeasurementPackage::LASER)
	{
		n_measurement = n_z_lidar_;
		NIS_p = &NIS_laser_;
	}
	else
	{
		n_measurement = n_z_radar_;
		NIS_p = &NIS_radar_;
	}

	// Calculate cross correlation matrix
	VectorXd factor1(n_x_), factor2(n_measurement);
	MatrixXd Tc (n_x_, n_measurement);

	Tc.setZero();

	for(int i=0; i<n_sig_; i++)
	{
	  factor1 = x_sigma_pred.col(i) - x_;
	  // Angle normalization
	  factor1(3) = NormalizeAngle(factor1(3));

	  factor2 = z_sig.col(i) - z_pred;
	  // Angle normalization
	  factor2(1) = NormalizeAngle(factor2(1));

	  Tc += weights_(i)*factor1*factor2.transpose();
	}


	// Calculate Kalman gain K;
	MatrixXd S_inv = S.inverse();
	MatrixXd K = Tc*S_inv;

	// Update state mean and covariance matrix
	VectorXd z_measured = meas_package.raw_measurements_;
	VectorXd z_err = z_measured-z_pred;
	z_err(1) = NormalizeAngle(z_err(1));

	x_ = x_ + K*z_err;
	x_(3) = NormalizeAngle(x_(3));

	P_ = P_ - K*S*K.transpose();

	// Compute NIS (Normalized Innovation Squared)
	*NIS_p = z_err.transpose()*S_inv*z_err;
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

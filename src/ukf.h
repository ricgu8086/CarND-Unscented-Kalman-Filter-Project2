#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF
{
public:

	///* initially set to false, set to true in first call of ProcessMeasurement
	bool is_initialized_;

	///* if this is false, laser measurements will be ignored (except for init)
	bool use_laser_;

	///* if this is false, radar measurements will be ignored (except for init)
	bool use_radar_;

	///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
	VectorXd x_;

	///* state covariance matrix
	MatrixXd P_;

	///* predicted sigma points matrix
	MatrixXd Xsig_pred_;

	///* time when the state is true, in us
	long long time_us_; // what the hell is this? the description is awful

	///* Previous timestamp in microseconds
	long long previous_timestamp_;

	///* Process noise standard deviation longitudinal acceleration in m/s^2
	double std_a_;

	///* Process noise standard deviation yaw acceleration in rad/s^2
	double std_yawdd_;

	///* Laser measurement noise standard deviation position1 in m
	double std_laspx_;

	///* Laser measurement noise standard deviation position2 in m
	double std_laspy_;

	///* Radar measurement noise standard deviation radius in m
	double std_radr_;

	///* Radar measurement noise standard deviation angle in rad
	double std_radphi_;

	///* Radar measurement noise standard deviation radius change in m/s
	double std_radrd_;

	///* Weights of sigma points
	VectorXd weights_;

	///* State dimension
	int n_x_;

	///* Augmented state dimension
	int n_aug_;

	///* Number of Sigma points
	int n_sig_;

	///* Dimension of Lidar measurement space
	int n_z_lidar_;

	///* Dimension of Radar measurement space
	int n_z_radar_;

	///* Sigma point spreading parameter
	double lambda_;

	///* Measurement noise for radar
	MatrixXd R_radar_;

	///* Measurement noise for lidar
	MatrixXd R_lidar_;

	///* the current NIS for radar
	double NIS_radar_;

	///* the current NIS for laser
	double NIS_laser_;

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
	void ProcessMeasurement(MeasurementPackage meas_package);

	/**
	 * Prediction predicts sigma points, the state and the state covariance
	 * matrix
	 * @param delta_t Time between k and k+1 in s
	 */
	MatrixXd Prediction(double delta_t);

	/**
	 * Generate the sigma points that represent the distribution
	 */
	MatrixXd GenerateSigmaPoints(void);

	/**
	 * Move the sigma points through the process model
	 */
	MatrixXd PredictSigmaPoints(MatrixXd Sigma_p, double delta_t);

	/**
	 * Apply the CTRV process model
	 */
	VectorXd ProcessModel(VectorXd SigmaPoint, double delta_t);

	/**
	 * Recover the approximate gaussian distribution from predicted sigma points
	 */
	void PredictMeanCovariance(MatrixXd Sigma_p_pred);

	/*
	 * Update estimates the upcoming measurement and incorporates the real measurement
	 * into the state and the state covariance matrix
	 */
	void Update(MatrixXd x_sigma_pred, MeasurementPackage meas_package);

	/**
	 * Convert the sigma points from the state space to the Radar measurement space.
	 */
	std::pair< std::pair<VectorXd, MatrixXd>, MatrixXd> \
		PredictMeasurementRadar(MatrixXd sigma_points);

	/**
	 * Convert the sigma points from the state space to the Lidar measurement space.
	 */
	std::pair< std::pair<VectorXd, MatrixXd>, MatrixXd> \
		PredictMeasurementLidar(MatrixXd sigma_points);

	/**
	 * Updates the state and the state covariance matrix using a measurement
	 * @param meas_package The measurement at k+1
	 */
	void UpdateState(std::pair< std::pair<VectorXd, MatrixXd>, MatrixXd> predicted, \
			MatrixXd x_sigma_pred, MeasurementPackage meas_package);

	/**
	 * Normalize an angle between -M_PI and M_PI
	 */
	double NormalizeAngle(double angle);
};

#endif /* UKF_H */

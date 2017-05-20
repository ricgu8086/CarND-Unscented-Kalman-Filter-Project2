#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools()
{
}

Tools::~Tools()
{
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
		const vector<VectorXd> &ground_truth)
{
	/**
	 * Calculate the RMSE here.
	 */
	int dim = estimations.at(0).size();
	VectorXd rmse(dim); // Some compilers doesn't allow this, other does..

	rmse.setZero();

	VectorXd residual;

	for(int i=0; i<estimations.size(); i++)
	{
		residual = estimations.at(i) - ground_truth.at(i);

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();

		rmse += residual;
	}

	// calculate the mean
	rmse /= estimations.size();

	// calculate the root
	rmse = rmse.array().sqrt();

	return rmse;
}

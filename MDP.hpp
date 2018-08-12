#ifndef MDP_HPP_
#define MDP_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include<map>

using namespace boost::numeric::ublas;


class MDP{

private :
	//mapping from action to probability transition matrix
	std::map<int,matrix<double> > actionTransitions;

	//matrix where entry (i,j) is the reward associated with taking action j from state j
	matrix<double> actionReward;

	//MDP discount factor in [0,1]
	double discount;

	//Total number of states in MPD
	int numStates;

	//Total number of actions in MDP
	int numActions;

public:

	MDP(std::map<int,matrix<double> > at, matrix<double> ar, double d);

	vector<double> policyReward(matrix<double> policy);

	matrix<double> policyTransitions(matrix<double> policy);

	vector<double> bellmanEquation(matrix<double> policyTrans, vector<double> policyRew, vector<double> valueFunc);

	vector<double> policyEvaluation(matrix<double> policyTrans, vector<double> policyRew, double epsilon);

	matrix<double> policyImprovement(vector<double> valueFunction);

	matrix<double> policyIteration();

};

#endif 

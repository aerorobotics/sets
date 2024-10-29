#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <memory>
#include <limits>

#include "../mdps/mdp.hpp"


class Solver {

    public:

        virtual ~Solver() {};

        virtual SolverResult solve(Eigen::Matrix<double,-1,1> state, int timestep, MDP* mdp, RNG& rng) {
            SolverResult result;
            return result; };

        // for uct
        virtual void set_max_depth(int max_depth) {throw std::logic_error("set_max_depth not implemented"); }
        virtual void set_N(int N) {throw std::logic_error("set_N not implemented"); }
        virtual void set_verbose(bool b) {throw std::logic_error("set_verbose not implemented"); }
        virtual void set_export_tree(bool b) {throw std::logic_error("set_export_tree not implemented"); }
        virtual void set_exploration_const(double c) {throw std::logic_error("set_exploration_const not implemented"); }
};



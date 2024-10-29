#pragma once 

#include <iterator>
#include <iostream>
#include <chrono>

#include "../mdps/mdp.hpp"
// #include "mdps/mdp.hpp"

#include "fd.hpp"
#include "dare.hpp"


class RNG {

    public: 

        void set_seed(unsigned int seed) { m_rng.seed(seed); }

        // Returns a permutation of [0, ..., n - 1].
        std::vector<int> permutation(int n) {
            std::vector<int> p(n);
            std::iota(p.begin(), p.end(), 0);
            std::shuffle(p.begin(), p.end(), m_rng);
            return p; }

        double uniform() { return m_dist(m_rng); }
        double gaussian() { return m_normal_dist(m_rng); }

    private:
        std::mt19937 m_rng;
        std::uniform_real_distribution<double> m_dist;
        // default is mean 0.0, std dev 1.0
        std::normal_distribution<double> m_normal_dist; 
    };


struct Trajectory {
    std::vector<Eigen::VectorXd> xs;
    std::vector<Eigen::VectorXd> us;
    std::vector<double> rs; 
    bool is_valid;
    double value; 
};


struct BranchData {
    Eigen::VectorXd x0;
    std::vector<std::vector<Eigen::VectorXd>> xbarss;
    std::vector<std::vector<Eigen::VectorXd>> ubarss;
    std::vector<std::vector<Eigen::VectorXd>> xss;
    std::vector<std::vector<Eigen::VectorXd>> uss;
    std::vector<std::vector<Eigen::MatrixXd>> dFdxss;
    std::vector<std::vector<Eigen::MatrixXd>> dFduss;  // (num iterations, horizon, matrix)
    std::vector<Eigen::MatrixXd> Cs; 
    std::vector<Eigen::VectorXd> eigenValuess;
    std::vector<Eigen::MatrixXd> eigenVectorss;
    std::vector<Eigen::VectorXd> delta_zs;
    std::vector<Eigen::VectorXd> delta_tilde_zs;
    std::vector<std::vector<Eigen::VectorXd>> delta_vss;
    std::vector<std::vector<Eigen::VectorXd>> delta_tilde_vss;
    std::vector<std::vector<double>> etas; // (num iterations, horizon)
    // profile (in number of ticks)
    std::vector<std::vector<int>> wctss; 
    std::vector<double> linearization_errors; 
};


struct SpecificBranchData
{
    bool is_valid;
    int branch_idx;
    Eigen::VectorXd delta_z_H;
    Eigen::VectorXd delta_z_H_unscaled;
    std::vector<Eigen::VectorXd> zbars;
    std::vector<Eigen::VectorXd> ubars;
    std::vector<Eigen::VectorXd> vs_ref;
    std::vector<Eigen::VectorXd> us_ref;
    std::vector<Eigen::VectorXd> zs_ref;
    std::vector<Eigen::VectorXd> us;
    std::vector<Eigen::VectorXd> xs;
    std::vector<double> rs;
};


struct CommonBranchData
{
    bool empty = true;
    // state_dim - 1 because we remove time from system
    double timestep0;
    Eigen::VectorXd zbar0;
    Eigen::VectorXd ubar0;
    Eigen::VectorXd vbar0;
    std::vector<Eigen::VectorXd> zbars;
    std::vector<Eigen::VectorXd> ubars;
    std::vector<Eigen::VectorXd> vbars;
    std::vector<Eigen::MatrixXd> As; // (state_dim-1, state_dim-1)
    std::vector<Eigen::MatrixXd> Bs; // (state_dim-1, action_dim)
    std::vector<Eigen::VectorXd> cs; // linear system shift : (state_dim, )
    std::vector<Eigen::MatrixXd> Ks; // gain matrix, (m_action_dim, m_state_dim-1)
    Eigen::MatrixXd C; // controllability matrix : (state_dim-1, action_dim * horizon)
    Eigen::MatrixXd C_pinv; // controllability matrix (action_dim * horizon, state_dim-1)
    Eigen::MatrixXd C_vbars_H; // C \bar{v}_{[H]} (m_state_dim-1,)
    Eigen::VectorXd eigenValues; // (state_dim-1,)
    Eigen::MatrixXd eigenValuesSqrt; // (state_dim-1,)
    Eigen::MatrixXd eigenVectors; // (state_dim-1,state_dim-1)
    Eigen::VectorXd eigenValues_to_search; // (state_dim-1,)
    Eigen::MatrixXd eigenValuesSqrt_to_search; // (state_dim-1,)
    Eigen::MatrixXd eigenVectors_to_search; // (state_dim-1,state_dim-1)
    // 
    Eigen::MatrixXd S;
    Eigen::MatrixXd S_inv;
    Eigen::VectorXd b;
    Eigen::VectorXd S_inv_b;
};


struct WallClockTimeData
{
    int tmp=0;
    int wct_dots_tmp = 0;
    
    bool cbd_empty;

    std::vector<std::chrono::time_point<std::chrono::system_clock>> breakpoints;
};


struct Tree {
    Eigen::VectorXd root;
    std::vector<Trajectory> trajs; 
    std::vector<CommonBranchData> cbds; 
    std::vector<SpecificBranchData> sbds; 
    std::vector<Eigen::VectorXd> node_states; 
    std::vector<std::tuple<double, int, int>> node_visit_statistics;
    Eigen::MatrixXd topology; 
};


struct AeroCoeffs {
    double C_D = 0.0;
    double C_L = 0.0;
    double C_M = 0.0;
    double C_Y = 0.0;
    double C_l = 0.0;
    double C_n = 0.0; 
};


struct SolverResult {
    bool success = false;
    // (for mpc) sometimes this is not the first element of best_us
    Trajectory mpc_traj;
    // (for planning) best trajectory 
    Trajectory planned_traj; 
    // (for vis)
    Tree tree; 
    std::vector<double> vs; 
};


Eigen::VectorXd sample_vec_from_cube(const Eigen::MatrixXd & cube, RNG& rng) {
    // cube in [n,2]
    Eigen::VectorXd vec(cube.rows()); 
    for (int ii=0; ii<cube.rows(); ii++) {
        double alpha = rng.uniform();
        vec[ii] = alpha * (cube(ii,1) - cube(ii,0)) + cube(ii,0);
    }
    return vec;
}


inline double square(double x) {
    return x * x;
}


bool is_vec_in_cube(const Eigen::VectorXd & vec, const Eigen::MatrixXd & cube) {
    return (vec.array() >= cube.col(0).array()).all() && (vec.array() <= cube.col(1).array()).all(); }


bool is_int_in_int_vec(int my_var, std::vector<int> my_list){
    bool found = (std::find(my_list.begin(), my_list.end(), my_var) != my_list.end());
    return found;
}


void extend_traj(Trajectory &t1, const Trajectory &t2) {
    t1.xs.insert( t1.xs.end(), t2.xs.begin(), t2.xs.end() );
    t1.us.insert( t1.us.end(), t2.us.begin(), t2.us.end() );
    t1.rs.insert( t1.rs.end(), t2.rs.begin(), t2.rs.end() ); 
    t1.is_valid = t1.is_valid && t2.is_valid;
    t1.value = t1.value + t2.value; }


void print_v(const std::vector<int> & v) {
    for (int ii = 0; ii < v.size(); ii++){
        std::cout << v[ii] << " "; 
    }
    std::cout << "" << std::endl; 
}


void print_v(const std::vector<double> & v) {
    for (int ii = 0; ii < v.size(); ii++){
        std::cout << v[ii] << " "; 
    }
    std::cout << "" << std::endl; 
}


void print_v(const Eigen::VectorXd & v) {
    for (int ii = 0; ii< v.size(); ii++){
        std::cout << v[ii] << " "; 
    }
    std::cout << "" << std::endl; 
}


void print_m(const Eigen::MatrixXd & m) {
    for (int jj = 0; jj < m.rows(); jj++) {
        for (int ii = 0; ii < m.cols(); ii++){
            std::cout << m.coeff(jj,ii) << " "; 
        }
        std::cout << "" << std::endl;
    }
    // std::cout << "" << std::endl; 
}


void print_vv(const std::vector<std::vector<double>> & vv) {
    for (auto v : vv) {
        print_v(v);
    } }


void print_vv(const std::vector<Eigen::VectorXd> & vv) {
    for (auto v : vv) {
        print_v(v);
    } }


void print_vm(const std::vector<Eigen::MatrixXd> & vm) {
    for (auto m : vm){
        print_m(m);
    } }


void print_vvv(const std::vector<std::vector<Eigen::VectorXd>> & vvv) {
    for (std::vector<Eigen::VectorXd> vv : vvv) {
        for (Eigen::VectorXd v : vv) {
            print_v(v);
        } 
    } } 


void print_m_specs(const Eigen::MatrixXd & m) {
    std::cout << "m.rows(): " << m.rows() << std::endl; 
    std::cout << "m.cols(): " << m.cols() << std::endl; 
    std::cout << "m.maxCoeff(): " << m.maxCoeff() << std::endl; 
    std::cout << "m.minCoeff(): " << m.minCoeff() << std::endl; 
}


void print_v_specs(const Eigen::VectorXd & v) {
    std::cout << "v.rows(): " << v.rows() << std::endl; 
    std::cout << "v.maxCoeff(): " << v.maxCoeff() << std::endl; 
    std::cout << "v.minCoeff(): " << v.minCoeff() << std::endl; 
}


void print_traj(const Trajectory & traj) {
    // at some points in time traj.xs is one length longer than us 
    std::vector<Eigen::VectorXd> xs = traj.xs;    
    if (xs.size() > traj.us.size()) {
        std::cout << "\tx0: "; print_v(xs[0]);
        xs.erase(xs.begin()); // remove first element
    }

    for (int ii = 0; ii < xs.size(); ii++){
        std::cout << "\tx: "; print_v(xs[ii]);
        std::cout << "\tu: "; print_v(traj.us[ii]);
        std::cout << "\tr: " << traj.rs[ii] << std::endl;
    } }


void print_shape_m(const Eigen::MatrixXd & m) {
    std::cout << m.rows() << ", " << m.cols() << std::endl;
} 


// Cartesion product of vector of vectors, adapted from: 
// https://stackoverflow.com/questions/5279051/how-can-i-create-cartesian-product-of-vector-of-vectors
void cart_product(
    std::vector<std::vector<double>>& rvvi,  // final result
    std::vector<double>&  rvi,   // current result 
    std::vector<std::vector<double>>::const_iterator me, // current input
    std::vector<std::vector<double>>::const_iterator end) // final input
{
    if(me == end) {
        // terminal condition of the recursion. We no longer have
        // any input vectors to manipulate. Add the current result (rvi)
        // to the total set of results (rvvvi).
        rvvi.push_back(rvi);
        return;
    }

    // need an easy name for my vector-of-ints
    const std::vector<double>& mevi = *me;
    for(std::vector<double>::const_iterator it = mevi.begin(); it != mevi.end(); it++) {
        // final rvi will look like "a, b, c, ME, d, e, f"
        // At the moment, rvi already has "a, b, c"
        rvi.push_back(*it);  // add ME
        cart_product(rvvi, rvi, me+1, end); // add "d, e, f"
        rvi.pop_back(); // clean ME off for next round
    }
}


std::vector<double> linspace(double start, double stop, int num_steps) {
    std::vector<double> line(num_steps); 
    double delta = (stop - start) / (num_steps - 1);
    for (int ii = 0; ii < num_steps; ii++){
        line[ii] = start + ii*delta;
    }
    return line; }


Eigen::VectorXd compute_x3(Eigen::VectorXd const & x1, Eigen::VectorXd const & x2, Eigen::MatrixXd const & cube) {
    // computes point of intersection between (i) line constructed from x1 and x2 and (ii) cube
    double eta = 0.0;
    for (int ii = 0; ii < x1.rows(); ii++) {
        if (x2(ii) - x1(ii) >= 0) {
            eta = std::max(eta, (x2(ii) - x1(ii)) / (cube(ii,1) - x1(ii)) );
        } else {
            eta = std::max(eta, (x2(ii) - x1(ii)) / (cube(ii,0) - x1(ii)) );
        }
    }
    Eigen::VectorXd x3;
    if (abs(eta) < 1.0e-12) {
        x3 = x1; 
    } else {
        x3 = (x2 - x1) / eta + x1; 
    }
    return x3;
}


double compute_alpha(Eigen::VectorXd const & x1, Eigen::VectorXd const & x2, Eigen::VectorXd const & x3) {
    // computes ratio of lengths between original line and line to intersection point 
    return (x2-x1).norm() / (x3-x1).norm();
}


Eigen::VectorXd scale_vec_in_cube(Eigen::VectorXd const & x1, Eigen::VectorXd const & x2, Eigen::MatrixXd const & cube) {
    Eigen::VectorXd x3 = compute_x3(x1, x2, cube);
    double alpha = compute_alpha(x1, x2, x3);
    double scale_factor = 0.7;
    Eigen::VectorXd x4 = x2;
    if (std::isnan(alpha) || alpha > scale_factor) {
        x4 = (x3 - x1) * scale_factor + x1;
    }
    
    // std::cout << "x1: "; print_v(x1);
    // std::cout << "x2: "; print_v(x2);
    // std::cout << "x3: "; print_v(x3);
    // std::cout << "x4: "; print_v(x4);

    return x4;
}


Eigen::MatrixXd scale_cube(Eigen::MatrixXd const & cube, double scale) {
    Eigen::MatrixXd scaled_cube(cube.rows(), 2);
    for (int ii=0; ii<cube.rows(); ii++) {
        scaled_cube(ii,0) = (cube(ii,0) + cube(ii,1))/2 - scale * (cube(ii,1) - cube(ii,0))/2.0;
        scaled_cube(ii,1) = (cube(ii,0) + cube(ii,1))/2 + scale * (cube(ii,1) - cube(ii,0))/2.0;
    }
    return scaled_cube;
}


Trajectory subsample_trajectory(Trajectory input_traj, int num_samples) {

    int traj_length = input_traj.xs.size();
    if (input_traj.xs.size() < num_samples) {
        return input_traj;
    }
    
    Trajectory subsampled_traj; 

    double step = static_cast<double>(traj_length - 1) / (num_samples - 1);
    // Linearly spaced indices
    for (size_t ii = 0; ii < num_samples; ii++) {
        size_t index = static_cast<size_t>(std::round(ii * step));
        subsampled_traj.xs.push_back(input_traj.xs[index]);
        subsampled_traj.us.push_back(input_traj.us[index]);
        subsampled_traj.rs.push_back(input_traj.rs[index]);
    }
    // always include last index 
    size_t index = traj_length-1;
    subsampled_traj.xs.push_back(input_traj.xs[index]);
    subsampled_traj.us.push_back(input_traj.us[index]);
    subsampled_traj.rs.push_back(input_traj.rs[index]);

    // update other things 
    subsampled_traj.is_valid = input_traj.is_valid;
    subsampled_traj.value = input_traj.value;
    return subsampled_traj;
}



// https://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

void removeRow(Eigen::VectorXd& vector, unsigned int rowToRemove)
{
    unsigned int numRows = vector.rows()-1;
    unsigned int numCols = 1;

    if( rowToRemove < numRows )
        vector.block(rowToRemove,0,numRows-rowToRemove,numCols) = vector.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    vector.conservativeResize(numRows,numCols);
}

void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}


std::vector<double> remove_nans_from_vec(std::vector<double> vec) {
    std::vector<double> vec_without_nans;
    for (int ii=0; ii<vec.size(); ii++){
        if (!std::isnan(vec[ii])) {
            vec_without_nans.push_back(vec[ii]);
        }
    }
    return vec_without_nans;
}


std::vector<double> remove_zeros_from_vec(std::vector<double> vec) {
    std::vector<double> vec_without_zeros;
    for (int ii=0; ii<vec.size(); ii++){
        if (vec[ii]!=0.0) {
            vec_without_zeros.push_back(vec[ii]);
        }
    }
    return vec_without_zeros;
}

// projects vector vec onto ellipse defined by axes and axis_lens
// axes is a matrix unit eigenvectors, where each vector is a column
// axis_len is the corresponding eigenvalue square root, where the ellipse's length along that axis is the value in axis_len
Eigen::VectorXd project_vector_onto_ellipse(Eigen::VectorXd vec, Eigen::MatrixXd axes, Eigen::VectorXd axis_len) {
    // coefficients
    Eigen::VectorXd alpha = axes.transpose() * vec;
    alpha = alpha.cwiseMax(-axis_len).cwiseMin(axis_len);
    // projected vector
    Eigen::VectorXd vec_proj = axes * alpha;
    // throw std::logic_error("Here!");
    return vec_proj;
}


// scales vector into ellipse, more conservative than above method, but maintains direction exactly, 
// I did not find this to be useful
Eigen::VectorXd project_vector_onto_ellipse2(Eigen::VectorXd vec, Eigen::MatrixXd axes, Eigen::VectorXd axis_len) {
    Eigen::VectorXd axis_len_sqrd_inv = (axis_len.cwiseProduct(axis_len)).cwiseInverse();
    Eigen::VectorXd y = axes.transpose() * vec;
    double denom = (y.transpose() * axis_len_sqrd_inv.matrix().asDiagonal() * y).value();
    double alpha = 1.0;
    if (denom > 0) {
        alpha = std::sqrt(1.0 / denom);
    }
    if (alpha > 1) {
        return vec;
    } else {
        return alpha * vec;
    }
}


// void DARE(Eigen::MatrixXd & X, const Eigen::MatrixXd & A, const Eigen::MatrixXd & A) {

// }


// Function to generate combinations recursively, written by ChatGPT3
void generateCombinationsHelper(const std::vector<std::vector<double>>& input,
                                std::vector<double>& currentCombination,
                                int currentVector,
                                std::vector<std::vector<double>>& result) {
    // If we have processed all vectors, add the current combination to the result
    if (currentVector == input.size()) {
        result.push_back(currentCombination);
        return;
    }

    // Iterate over the current vector and include each element in the combination
    for (double element : input[currentVector]) {
        // Include the current element in the combination
        currentCombination.push_back(element);

        // Recursively generate combinations for the remaining vectors
        generateCombinationsHelper(input, currentCombination, currentVector + 1, result);

        // Remove the last element to backtrack and try the next element
        currentCombination.pop_back();
    }
}

// Function to generate all combinations of a vector of vectors of doubles
std::vector<std::vector<double>> generateCombinations(const std::vector<std::vector<double>>& input) {
    std::vector<std::vector<double>> result;
    std::vector<double> currentCombination;

    // Start generating combinations from the first vector
    generateCombinationsHelper(input, currentCombination, 0, result);

    return result;
}

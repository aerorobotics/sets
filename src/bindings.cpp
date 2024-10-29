
#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "mdps/mdp.hpp"
#include "mdps/dots.hpp"

#include "mdps/sixdofaircraft.hpp"
#include "mdps/game_sixdofaircraft.hpp"

#include "solvers/solver.hpp"
#include "solvers/uct.hpp"
#include "solvers/uct2.hpp"
// #include "solvers/scp.hpp"
// #include "solvers/scp2.hpp"
#include "solvers/uniform_discretization_mcts.hpp"
#include "solvers/double_progressive_widening_mcts.hpp"
#include "solvers/uniform_discretization_predictive_sampling.hpp"
#include "solvers/dots_predictive_sampling.hpp"

#include "util/util.hpp"


MDP* get_mdp(std::string mdp_name, std::string config_path) {
    YAML::Node config = YAML::LoadFile(config_path);    
    MDP* mdp;
    if (mdp_name == "SixDOFAircraft") {
        mdp = new SixDOFAircraft(config_path);
    } else if (mdp_name == "GameSixDOFAircraft") {
        mdp = new GameSixDOFAircraft(config_path);
    } else {
        std::cout << "Unknown system: " << mdp_name << std::endl;
        exit(1); }
    return mdp; 
}


DOTS* get_dots_mdp() {
    DOTS* dots_mdp = new DOTS();
    return dots_mdp;
}


UCT* get_uct() {
    UCT* uct;
    uct = new UCT();
    return uct;
}


UCT2* get_uct2() {
    UCT2* uct2;
    uct2 = new UCT2();
    return uct2;
}


UniformDiscretization_MCTS* get_ud_mcts() {
    UniformDiscretization_MCTS* ud_mcts;
    ud_mcts = new UniformDiscretization_MCTS();
    return ud_mcts;
}


DPW_MCTS* get_dpw_mcts() {
    DPW_MCTS* dpw_mcts;
    dpw_mcts = new DPW_MCTS();
    return dpw_mcts;
}


UD_PS* get_ud_ps() {
    UD_PS* ud_ps;
    ud_ps = new UD_PS();
    return ud_ps;
}


SE_PS* get_se_ps() {
    SE_PS* se_ps;
    se_ps = new SE_PS();
    return se_ps;
}


SolverResult run_uct(DOTS* dots_mdp, UCT* uct, Eigen::VectorXd curr_state, RNG& rng) {
    // solve 
    SolverResult r = uct->solve(curr_state, curr_state(dots_mdp->timestep_idx()), static_cast<MDP*>(dots_mdp), rng);
    return r; 
}


SolverResult run_uct2(DOTS* dots_mdp, UCT2* uct2, Eigen::VectorXd curr_state, RNG& rng) {
    // solve 
    SolverResult r = uct2->solve(curr_state, curr_state(dots_mdp->timestep_idx()), static_cast<MDP*>(dots_mdp), rng);
    return r; 
}

SolverResult run_ud_mcts(MDP* ground_mdp, UniformDiscretization_MCTS* ud_mcts, Eigen::VectorXd curr_state, RNG& rng) {
    // solve 
    SolverResult r = ud_mcts->solve(curr_state, curr_state(ground_mdp->timestep_idx()), ground_mdp, rng);
    return r; 
}


SolverResult run_dpw_mcts(MDP* ground_mdp, DPW_MCTS* dpw_mcts, Eigen::VectorXd curr_state, RNG& rng) {
    // solve 
    SolverResult r = dpw_mcts->solve(curr_state, curr_state(ground_mdp->timestep_idx()), ground_mdp, rng);
    return r; 
}


SolverResult run_ud_ps(MDP* ground_mdp, UD_PS* ud_ps, Eigen::VectorXd curr_state, RNG& rng) {
    // solve 
    SolverResult r = ud_ps->solve(curr_state, curr_state(ground_mdp->timestep_idx()), ground_mdp, rng);
    return r; 
}

SolverResult run_se_ps(DOTS* dots_mdp, SE_PS* se_ps, Eigen::VectorXd curr_state, RNG& rng) {
    // solve 
    SolverResult r = se_ps->solve(curr_state, curr_state(dots_mdp->timestep_idx()), static_cast<MDP*>(dots_mdp), rng);
    return r; 
}


// SolverResult run_scp(MDP* mdp, SCP* scp, Eigen::VectorXd curr_state, RNG& rng) {
//     // solve 
//     SolverResult r = scp->solve(curr_state, curr_state(mdp->timestep_idx()), mdp, rng);
//     return r; 
// }


// SolverResult run_scp2(MDP* mdp, SCP2* scp2, Eigen::VectorXd curr_state, int non_augmented_state_dim, RNG& rng) {
//     // solve 
//     SolverResult r = scp2->solve2(curr_state, curr_state(mdp->timestep_idx()), non_augmented_state_dim, mdp, rng);
//     return r; 
// }


// Eigen::Matrix<double,6,1> wrapper_aero_model(const Eigen::VectorXd & state, const Eigen::VectorXd & action, MDP* mdp) {
//     if (mdp->name() == "SixDOFAircraft") {
//         return static_cast<SixDOFAircraft*>(mdp)->aero_model(state, action);
//     } else if (mdp->name() == "GameSixDOFAircraft") {
//         return static_cast<GameSixDOFAircraft*>(mdp)->aero_model(state, action);
//     } else {
//         std::cout << "Bad mdp to call wrapper_aero_model on: " << mdp->name() << std::endl;
//         throw std::logic_error("Unknown mdp!");
//     }
// }


// std::array<double,6> wrapper_aero_coeffs(const Eigen::VectorXd & state, const Eigen::VectorXd & action, MDP* mdp) {
//     return static_cast<SixDOFAircraft*>(mdp)->compute_aero_coeffs(state, action);
// }


// python interface
PYBIND11_MODULE(bindings, m) {
    
    m.def("get_mdp", &get_mdp, "get_mdp");
    m.def("get_dots_mdp", &get_dots_mdp, "get_dots_mdp");
    m.def("get_uct", &get_uct, "get_uct");
    m.def("get_uct2", &get_uct2, "get_uct2");

    // m.def("wrapper_compute_aero_forces_and_moments_from_state_diff", &wrapper_compute_aero_forces_and_moments_from_state_diff, "wrapper_compute_aero_forces_and_moments_from_state_diff");
    // m.def("wrapper_aero_model", &wrapper_aero_model, "wrapper_aero_model");
    // m.def("wrapper_aero_coeffs", &wrapper_aero_coeffs, "wrapper_aero_coeffs");

    // m.def("run_solver", &run_solver, "run_solver");
    m.def("rollout_action_sequence", &rollout_action_sequence, "rollout_action_sequence");
    m.def("run_uct", &run_uct, "run_uct");
    m.def("run_uct2", &run_uct2, "run_uct2");
    // m.def("run_scp", &run_scp, "run_scp");
    // m.def("run_scp2", &run_scp2, "run_scp2");

    m.def("get_ud_mcts", &get_ud_mcts, "get_ud_mcts");
    m.def("run_ud_mcts", &run_ud_mcts, "run_ud_mcts");

    m.def("get_dpw_mcts", &get_dpw_mcts, "get_dpw_mcts");
    m.def("run_dpw_mcts", &run_dpw_mcts, "run_dpw_mcts");

    m.def("get_ud_ps", &get_ud_ps, "get_ud_ps");
    m.def("run_ud_ps", &run_ud_ps, "run_ud_ps");

    m.def("get_se_ps", &get_se_ps, "get_se_ps");
    m.def("run_se_ps", &run_se_ps, "run_se_ps");

    pybind11::class_<RNG> (m, "RNG")
        .def(pybind11::init())
        .def("set_seed", &RNG::set_seed);

    pybind11::class_<DOTS> (m, "DOTS")
        .def(pybind11::init())
        .def("update_augmented_state_only", &DOTS::update_augmented_state_only)
        .def("R", &DOTS::R)
        .def("empty_control", &DOTS::empty_control)
        .def("add_obstacle", &DOTS::add_obstacle)
        .def("clear_obstacles", &DOTS::clear_obstacles)
        .def("add_thermal", &DOTS::add_thermal)
        .def("clear_thermals", &DOTS::clear_thermals)
        .def("add_target", &DOTS::add_target)
        .def("clear_targets", &DOTS::clear_targets)
        .def("set_param", &DOTS::set_param);

    pybind11::class_<MDP> (m, "MDP")
        .def(pybind11::init())
        .def("X", &MDP::X)
        .def("U", &MDP::U)
        .def("F", &MDP::F)
        .def("R", &MDP::R)
        .def("R_verbose", &MDP::R_verbose)
        .def("H", &MDP::H)
        .def("dFdx", &MDP::dFdx)
        .def("dFdu", &MDP::dFdu)
        .def("set_xd", &MDP::set_xd)
        .def("set_x0", &MDP::set_x0)
        .def("set_dt", &MDP::set_dt)
        .def("is_state_valid", &MDP::is_state_valid)
        .def("dt", &MDP::dt)
        .def("timestep_idx", &MDP::timestep_idx)
        .def("obstacles", &MDP::obstacles)
        .def("add_obstacle", &MDP::add_obstacle)
        .def("clear_obstacles", &MDP::clear_obstacles)
        .def("add_target", &MDP::add_target)
        .def("clear_targets", &MDP::clear_targets)
        .def("add_thermal", &MDP::add_thermal)
        .def("clear_thermals", &MDP::clear_thermals)
        .def("set_weights", &MDP::set_weights)
        .def("eval_ff", &MDP::eval_ff)
        .def("sample_state", &MDP::sample_state)
        .def("initial_state", &MDP::initial_state)
        .def("empty_control", &MDP::empty_control)
        .def("state_dim", &MDP::state_dim)
        .def("action_dim", &MDP::action_dim)
        .def("B", &MDP::B)
        .def("F_non_augmented", &MDP::F_non_augmented)
        .def("F_timeless", &MDP::F_timeless)
        .def("set_trajs", &MDP::set_trajs)
        .def("add_traj", &MDP::add_traj);

    pybind11::class_<UCT> (m, "UCT")
        .def(pybind11::init())
        .def("set_param", &UCT::set_param);

    pybind11::class_<UCT2> (m, "UCT2")
        .def(pybind11::init())
        .def("set_param", &UCT2::set_param);

    pybind11::class_<UniformDiscretization_MCTS> (m, "UniformDiscretization_MCTS")
        .def(pybind11::init())
        .def("set_param", &UniformDiscretization_MCTS::set_param);

    pybind11::class_<DPW_MCTS> (m, "DPW_MCTS")
        .def(pybind11::init())
        .def("set_param", &DPW_MCTS::set_param);

    pybind11::class_<UD_PS> (m, "UD_PS")
        .def(pybind11::init())
        .def("set_param", &UD_PS::set_param);

    pybind11::class_<SE_PS> (m, "SE_PS")
        .def(pybind11::init())
        .def("set_param", &SE_PS::set_param);

    // pybind11::class_<SCP> (m, "SCP")
    //     .def(pybind11::init())
    //     .def("set_horizon", &SCP::set_horizon)
    //     .def("set_initial_guess_mode", &SCP::set_initial_guess_mode)
    //     .def("set_initial_guess", &SCP::set_initial_guess)
    //     .def("set_param", &SCP::set_param);

    // pybind11::class_<SCP2> (m, "SCP2")
    //     .def(pybind11::init())
    //     .def("set_weights", &SCP2::set_weights)
    //     .def("set_horizon", &SCP2::set_horizon)
    //     .def("set_initial_guess_mode", &SCP2::set_initial_guess_mode)
    //     .def("set_initial_guess", &SCP2::set_initial_guess)
    //     .def("set_param", &SCP2::set_param);

    pybind11::class_<Solver> (m, "Solver")
        .def(pybind11::init())
        .def("set_max_depth", &Solver::set_max_depth)
        .def("set_verbose", &Solver::set_verbose)
        .def("set_exploration_const", &Solver::set_exploration_const)
        .def("set_export_tree", &Solver::set_export_tree)
        .def("set_N", &Solver::set_N);

    pybind11::class_<SolverResult> (m, "SolverResult")
        .def(pybind11::init())
        .def_readwrite("vs", &SolverResult::vs)
        .def_readwrite("tree", &SolverResult::tree)
        .def_readwrite("mpc_traj", &SolverResult::mpc_traj)
        .def_readwrite("planned_traj", &SolverResult::planned_traj)
        .def_readwrite("mpc_traj", &SolverResult::mpc_traj)
        .def_readwrite("success", &SolverResult::success);

    pybind11::class_<Trajectory> (m, "Trajectory")
        .def(pybind11::init())
        .def_readwrite("is_valid", &Trajectory::is_valid)
        .def_readwrite("value", &Trajectory::value)
        .def_readwrite("xs", &Trajectory::xs)
        .def_readwrite("us", &Trajectory::us)
        .def_readwrite("rs", &Trajectory::rs);

    pybind11::class_<AeroCoeffs> (m, "AeroCoeffs")
        .def(pybind11::init())
        .def_readwrite("C_D", &AeroCoeffs::C_D)
        .def_readwrite("C_L", &AeroCoeffs::C_L)
        .def_readwrite("C_M", &AeroCoeffs::C_M)
        .def_readwrite("C_Y", &AeroCoeffs::C_Y)
        .def_readwrite("C_l", &AeroCoeffs::C_l)
        .def_readwrite("C_n", &AeroCoeffs::C_n);

    pybind11::class_<Tree> (m, "Tree")
        .def(pybind11::init())
        .def_readwrite("node_states", &Tree::node_states)
        .def_readwrite("cbds", &Tree::cbds)
        .def_readwrite("sbds", &Tree::sbds)
        .def_readwrite("trajs", &Tree::trajs)
        .def_readwrite("topology", &Tree::topology)
        .def_readwrite("node_visit_statistics", &Tree::node_visit_statistics)
        .def_readwrite("root", &Tree::root);

    pybind11::class_<BranchData> (m, "BranchData")
        .def(pybind11::init())
        .def_readwrite("x0", &BranchData::x0)
        .def_readwrite("xbarss", &BranchData::xbarss)
        .def_readwrite("ubarss", &BranchData::ubarss)
        .def_readwrite("dFdxss", &BranchData::dFdxss)
        .def_readwrite("dFduss", &BranchData::dFduss)
        .def_readwrite("Cs", &BranchData::Cs)
        .def_readwrite("eigenValuess", &BranchData::eigenValuess)
        .def_readwrite("eigenVectorss", &BranchData::eigenVectorss)
        .def_readwrite("delta_zs", &BranchData::delta_zs)
        .def_readwrite("delta_vss", &BranchData::delta_vss)
        .def_readwrite("etas", &BranchData::etas)
        .def_readwrite("delta_tilde_zs", &BranchData::delta_tilde_zs)
        .def_readwrite("delta_tilde_vss", &BranchData::delta_tilde_vss)
        .def_readwrite("xss", &BranchData::xss)
        .def_readwrite("uss", &BranchData::uss)
        .def_readwrite("linearization_errors", &BranchData::linearization_errors)
        .def_readwrite("wctss", &BranchData::wctss);

    pybind11::class_<SpecificBranchData> (m, "SpecificBranchData")
        .def(pybind11::init())
        .def_readwrite("branch_idx", &SpecificBranchData::branch_idx)
        .def_readwrite("is_valid", &SpecificBranchData::is_valid)
        .def_readwrite("delta_z_H", &SpecificBranchData::delta_z_H)
        .def_readwrite("delta_z_H_unscaled", &SpecificBranchData::delta_z_H_unscaled)
        .def_readwrite("zbars", &SpecificBranchData::zbars)
        .def_readwrite("ubars", &SpecificBranchData::ubars)
        .def_readwrite("vs_ref", &SpecificBranchData::vs_ref)
        .def_readwrite("us_ref", &SpecificBranchData::us_ref)
        .def_readwrite("zs_ref", &SpecificBranchData::zs_ref)
        .def_readwrite("us", &SpecificBranchData::us)
        .def_readwrite("rs", &SpecificBranchData::rs)
        .def_readwrite("xs", &SpecificBranchData::xs);

    pybind11::class_<CommonBranchData> (m, "CommonBranchData")
        .def(pybind11::init())
        .def_readwrite("empty", &CommonBranchData::empty)
        .def_readwrite("timestep0", &CommonBranchData::timestep0)
        .def_readwrite("zbars", &CommonBranchData::zbars)
        .def_readwrite("ubars", &CommonBranchData::ubars)
        .def_readwrite("As", &CommonBranchData::As)
        .def_readwrite("Bs", &CommonBranchData::Bs)
        .def_readwrite("cs", &CommonBranchData::cs)
        .def_readwrite("C", &CommonBranchData::C)
        .def_readwrite("C_pinv", &CommonBranchData::C_pinv)
        .def_readwrite("eigenValues", &CommonBranchData::eigenValues)
        .def_readwrite("eigenVectors", &CommonBranchData::eigenVectors)
        .def_readwrite("S", &CommonBranchData::S)
        .def_readwrite("S_inv", &CommonBranchData::S_inv)
        .def_readwrite("b", &CommonBranchData::b)
        .def_readwrite("S_inv_b", &CommonBranchData::S_inv_b);

    pybind11::class_<WallClockTimeData> (m, "WallClockTimeData")
        .def(pybind11::init())
        .def_readwrite("wct_dots_tmp", &WallClockTimeData::wct_dots_tmp);

}


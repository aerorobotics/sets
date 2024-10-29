

#pragma once

#define _DOTS_HPP_PROFILING_ON_ 0

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "mdp.hpp"
#include "sixdofaircraft.hpp"
#include "../util/util.hpp"


class DOTS : public MDP {

    public:

        DOTS() { }


        void set_param(MDP* mdp, 
                       std::string expansion_mode, 
                       std::string initialize_mode, 
                       std::vector<std::string> special_actions, 
                       int num_branches, 
                       int decision_making_horizon, 
                       int dynamics_horizon, 
                       std::string spectral_branches_mode, 
                       std::string control_mode, 
                       std::string scale_mode, 
                       std::string damping_mode, 
                       std::vector<double> modal_damping_gains, 
                       Eigen::MatrixXd rho, 
                       double greedy_gain, 
                       double greedy_rate, 
                       double greedy_min_dist, 
                       bool baseline_mode_on,
                       int num_action_discretization,
                       bool verbose) {

            if (expansion_mode == "SpectralSearch") {
                m_expansion_mode = 0;
            } else {
                throw std::logic_error("dots expansion_mode not recognized");
            }

            if (initialize_mode == "empty_control") {
                m_initialize_mode = 0;    
            } else if (initialize_mode == "prev_control") {
                m_initialize_mode = 1; 
            } else if (initialize_mode == "thermal_compensation") {
                m_initialize_mode = 2;
            } else if (initialize_mode == "stabilization") {
                m_initialize_mode = 3;
            } else {
                throw std::logic_error("dots initialize_mode not recognized");
            }

            if (spectral_branches_mode == "decreasing") {
                m_spectral_branches_mode = 0; 
            } else if (spectral_branches_mode == "generalized_velocities") {
                m_spectral_branches_mode = 1;
            } else if (spectral_branches_mode == "generalized_positions") {
                m_spectral_branches_mode = 2;
            } else if (spectral_branches_mode == "my_idxs") {
                m_spectral_branches_mode = 3;
            } else {
                throw std::logic_error("dots spectral_branches_mode not recognized");
            }

            if (control_mode == "open_loop") {
                m_control_mode = 0; 
            } else if (control_mode == "closed_loop_discrete") {
                m_control_mode = 1; 
            } else if (control_mode == "closed_loop_continuous") {
                m_control_mode = 2; 
            } else {
                throw std::logic_error("dots control_mode not recognized");
            }

            if (scale_mode == "empty") {
                m_scale_mode = 0;
            } else if (scale_mode == "absolute_clip") {
                m_scale_mode = 1;
            } else if (scale_mode == "displacement_clip") {
                m_scale_mode = 2;
            } else if (scale_mode == "displacement_uniform") {
                m_scale_mode = 3;
            } else {
                throw std::logic_error("dots scale_mode not recognized");
            }

            if (damping_mode == "damp_terminal_state") {
                m_damping_mode = 0;
            } else if (damping_mode == "damp_displacement_between_systems") {
                m_damping_mode = 1;
            } else if (damping_mode == "damp_displacement_in_time") {
                m_damping_mode = 2;
            } else {
                throw std::logic_error("dots damping mode not recognized");
            }

            // if (discrete_baseline_mode_on) {
            //     m_discrete_baseline_mode_on = true;
            //     m_num_action_discretization = num_action_discretization;
            // }

            m_ground_mdp = mdp; 
            m_max_num_branches = num_branches;
            m_decision_making_horizon = decision_making_horizon;
            m_dynamics_horizon = dynamics_horizon;
            m_action_dim = m_ground_mdp->action_dim();
            m_rho = rho;
            m_verbose = verbose; 
            m_damping_gains = modal_damping_gains;
            m_greedy_rate = greedy_rate;
            m_greedy_gain = greedy_gain;
            m_greedy_min_dist = greedy_min_dist;
            
            m_state_dim = 0;
            if (m_ground_mdp->name() == "GameSixDOFAircraft") {
                m_state_dim = m_ground_mdp->not_augmented_state_dim(); 
            } else {
                m_state_dim = m_ground_mdp->state_dim(); 
            }
            
            m_empty_action_on = std::find(std::begin(special_actions), std::end(special_actions), "empty") != std::end(special_actions);
            m_greedy_action_on = std::find(std::begin(special_actions), std::end(special_actions), "greedy") != std::end(special_actions);
            m_greedy_action2_on = std::find(std::begin(special_actions), std::end(special_actions), "greedy2") != std::end(special_actions);
            m_greedy_action3_on = std::find(std::begin(special_actions), std::end(special_actions), "greedy3") != std::end(special_actions);
            m_equ_action_on = std::find(std::begin(special_actions), std::end(special_actions), "equillibrium") != std::end(special_actions);
            m_max_num_special_actions = 5; // reserve first 5 indices for special actions 

            int spectral_state_dim = m_ground_mdp->state_dim()-1;
            if (m_ground_mdp->name()=="GameSixDOFAircraft"){
                spectral_state_dim = m_ground_mdp->not_augmented_state_dim()-1;
            }
            m_num_spectral_actions = 2 * (spectral_state_dim); 
            if (m_spectral_branches_mode == 1) {
                m_num_spectral_actions = 2 * m_ground_mdp->velocity_idxs().size();
            } else if (m_spectral_branches_mode == 2) {
                m_num_spectral_actions = 2 * m_ground_mdp->position_idxs().size();
            } else if (m_spectral_branches_mode == 3) {
                m_num_spectral_actions = 2 * m_ground_mdp->my_idxs().size();
            }

            Eigen::MatrixXd U = m_ground_mdp->U();
            for (int ii=0; ii<m_action_dim; ii++){
                double du = U(ii,1) - U(ii,0);
                if (std::abs(du) < 1e-12) {
                    throw std::logic_error("control space not expected");
                }
            }
            
            if (m_verbose) {
                std::cout << "m_ground_mdp: " << m_ground_mdp << std::endl;
                std::cout << "m_expansion_mode: " << m_expansion_mode << std::endl;
                std::cout << "m_decision_making_horizon: " << m_decision_making_horizon << std::endl;
                std::cout << "m_dynamics_horizon: " << m_dynamics_horizon << std::endl;
                std::cout << "m_state_dim: " << m_state_dim << std::endl;
                std::cout << "m_action_dim: " << m_action_dim << std::endl;
                std::cout << "m_verbose: " << m_verbose << std::endl;
                std::cout << "m_empty_action_on: " << m_empty_action_on << std::endl;
                std::cout << "m_greedy_action_on: " << m_greedy_action_on << std::endl;
                std::cout << "m_equ_action_on: " << m_equ_action_on << std::endl;
            }
        }

        void expand_ii_mem_safe(RNG& rng, int branch_idx, const Eigen::VectorXd & prev_state, 
            const Eigen::VectorXd & prev_control, Trajectory & traj) override {
            
            CommonBranchData cbd;
            SpecificBranchData sbd;
            WallClockTimeData wctd;
            expand_ii(rng, branch_idx, prev_state, prev_control, 
                traj, 
                cbd, 
                sbd, 
                wctd);
            alloc_cbd(cbd, 0);
            alloc_sbd(sbd, 0);
        }


        void expand_ii(RNG& rng, int branch_idx, const Eigen::VectorXd & prev_state, 
            const Eigen::VectorXd & prev_control, Trajectory & traj, CommonBranchData & reuse_cbd, 
            SpecificBranchData & sbd, WallClockTimeData & wctdata) override {

            if (m_decision_making_horizon % m_dynamics_horizon != 0) {
                throw std::logic_error("horizons should be clean ratio");
            }

            int num_steps = int(m_decision_making_horizon / m_dynamics_horizon);

            CommonBranchData cbd; 
            alloc_cbd(cbd, m_decision_making_horizon);
            alloc_sbd(sbd, m_decision_making_horizon);
            // alloc_traj(traj, m_decision_making_horizon);
            sbd.is_valid = true;

            for (int kk=0; kk<num_steps; kk++) {

                CommonBranchData cbd_kk;
                SpecificBranchData sbd_kk;
                alloc_cbd(cbd_kk, m_dynamics_horizon);
                alloc_sbd(sbd_kk, m_dynamics_horizon);

                Eigen::VectorXd x0;
                Eigen::VectorXd u0;
                if (kk==0) {
                    if (!reuse_cbd.empty) {
                        copy_partial_cbd(cbd_kk, reuse_cbd, 0, m_dynamics_horizon);
                    }
                    x0 = prev_state;
                    u0 = prev_control;
                } else {
                    x0 = sbd.xs[kk*m_dynamics_horizon-1];
                    u0 = sbd.us[kk*m_dynamics_horizon-1];
                }
                bool is_valid = expand_ii_once(rng, branch_idx, x0, u0, kk, cbd_kk, sbd_kk, wctdata);

                if (kk==0 && reuse_cbd.empty) {
                    alloc_cbd(reuse_cbd, m_dynamics_horizon);
                    copy_partial_cbd(reuse_cbd, cbd_kk, 0, m_dynamics_horizon);
                    reuse_cbd.timestep0 = prev_state(m_ground_mdp->timestep_idx());
                }

                copy_partial_cbd(cbd, cbd_kk, kk*m_dynamics_horizon, (kk+1)*m_dynamics_horizon);
                copy_partial_sbd(sbd, sbd_kk, cbd_kk, kk*m_dynamics_horizon, (kk+1)*m_dynamics_horizon);

                traj.xs.insert(traj.xs.end(),sbd_kk.xs.begin(),sbd_kk.xs.end());
                traj.us.insert(traj.us.end(),sbd_kk.us.begin(),sbd_kk.us.end());
                traj.rs.insert(traj.rs.end(),sbd_kk.rs.begin(),sbd_kk.rs.end());
                traj.is_valid = is_valid;

                if (!is_valid) { 
                    traj.xs = std::vector<Eigen::VectorXd> (1,prev_state);
                    traj.us = std::vector<Eigen::VectorXd> (1,prev_control);
                    traj.is_valid = false;
                    sbd.is_valid = false;
                    break;
                }
            }
            if (traj.is_valid) {
                traj.value = std::accumulate(traj.rs.begin(), traj.rs.end(), 0.0) / traj.xs.size();
            }
        }


        void alloc_cbd(CommonBranchData & cbd, int list_size) {
            cbd.ubar0 = Eigen::VectorXd::Zero(m_action_dim);
            cbd.zbars.resize(list_size);
            cbd.ubars.resize(list_size);
            cbd.vbars.resize(list_size);
            cbd.As.resize(list_size);
            cbd.Bs.resize(list_size);
            cbd.cs.resize(list_size);
            cbd.Ks.resize(list_size);
            cbd.C = Eigen::MatrixXd::Zero(m_state_dim-1, m_action_dim * m_dynamics_horizon); 
            cbd.C_pinv = Eigen::MatrixXd::Zero(m_action_dim * m_dynamics_horizon, m_state_dim-1);                 
            cbd.C_vbars_H = Eigen::VectorXd::Zero(m_state_dim-1); 
            cbd.eigenValues = Eigen::VectorXd::Zero(m_state_dim-1);
            cbd.eigenValuesSqrt = Eigen::VectorXd::Zero(m_state_dim-1);
            cbd.eigenVectors = Eigen::MatrixXd::Zero(m_state_dim-1, m_state_dim-1);
            cbd.S = Eigen::MatrixXd::Zero(m_action_dim,m_action_dim);
            cbd.S_inv = Eigen::MatrixXd::Zero(m_action_dim,m_action_dim);
            cbd.b = Eigen::VectorXd::Zero(m_action_dim);
        }


        // copy the first (timestep_end - timestep_start) idxs from cbd2 into cbd1's list
        void copy_partial_cbd(CommonBranchData & cbd1, CommonBranchData & cbd2, int timestep_start, int timestep_end) {
            cbd1.ubar0 = cbd2.ubar0;
            cbd1.C = cbd2.C;
            cbd1.C_pinv = cbd2.C_pinv;
            cbd1.C_vbars_H = cbd2.C_vbars_H;
            cbd1.eigenValues = cbd2.eigenValues;
            cbd1.eigenValuesSqrt = cbd2.eigenValuesSqrt;
            cbd1.eigenVectors = cbd2.eigenVectors;
            cbd1.S = cbd2.S;
            cbd1.S_inv = cbd2.S_inv;
            cbd1.b = cbd2.b;
            for (int kk=timestep_start; kk<timestep_end; kk++){
                cbd1.zbars[kk] = cbd2.zbars[kk-timestep_start];
                cbd1.ubars[kk] = cbd2.ubars[kk-timestep_start];
                cbd1.vbars[kk] = cbd2.vbars[kk-timestep_start];
                cbd1.As[kk] = cbd2.As[kk-timestep_start];
                cbd1.Bs[kk] = cbd2.Bs[kk-timestep_start];
                cbd1.cs[kk] = cbd2.cs[kk-timestep_start];
                cbd1.Ks[kk] = cbd2.Ks[kk-timestep_start];
            }
        }


        void alloc_sbd(SpecificBranchData & sbd, int list_size) {
            // specific alloc 
            sbd.delta_z_H = Eigen::VectorXd::Zero(m_state_dim-1);
            sbd.delta_z_H_unscaled = Eigen::VectorXd::Zero(m_state_dim-1);
            sbd.zbars.resize(list_size, Eigen::VectorXd::Zero(m_state_dim-1)); // std vector of Eigen::VectorXd
            sbd.ubars.resize(list_size, Eigen::VectorXd::Zero(m_action_dim)); 
            sbd.vs_ref.resize(list_size, Eigen::VectorXd::Zero(m_action_dim)); 
            sbd.us_ref.resize(list_size, Eigen::VectorXd::Zero(m_action_dim)); 
            sbd.zs_ref.resize(list_size, Eigen::VectorXd::Zero(m_state_dim-1));
            sbd.us.resize(list_size, Eigen::VectorXd::Zero(m_action_dim)); 
            sbd.xs.resize(list_size, Eigen::VectorXd::Zero(m_ground_mdp->state_dim())); 
            sbd.rs.resize(list_size);
        }


        void alloc_traj(Trajectory & traj, int list_size) {
            traj.xs.resize(list_size, Eigen::VectorXd::Zero(m_ground_mdp->state_dim())); 
            traj.us.resize(list_size, Eigen::VectorXd::Zero(m_action_dim)); 
            traj.rs.resize(list_size); 
        }


        void copy_partial_sbd(SpecificBranchData & sbd1, SpecificBranchData & sbd2, 
                              CommonBranchData & cbd2, int timestep_start, int timestep_end) {
            // specific alloc 
            sbd1.delta_z_H = sbd2.delta_z_H;
            sbd1.delta_z_H_unscaled = sbd2.delta_z_H_unscaled;
            for (int kk=timestep_start; kk<timestep_end; kk++){
                sbd1.zbars[kk] = cbd2.zbars[kk-timestep_start];
                sbd1.ubars[kk] = cbd2.ubars[kk-timestep_start];
                sbd1.vs_ref[kk] = sbd2.vs_ref[kk-timestep_start];
                sbd1.us_ref[kk] = sbd2.us_ref[kk-timestep_start];
                sbd1.zs_ref[kk] = sbd2.zs_ref[kk-timestep_start];
                sbd1.us[kk] = sbd2.us[kk-timestep_start];
                sbd1.xs[kk] = sbd2.xs[kk-timestep_start];
                sbd1.rs[kk] = sbd2.rs[kk-timestep_start];
            }
        }


        // bool expand_ii_once_baseline(RNG& rng, int branch_idx, const Eigen::VectorXd & prev_state, 
        //     const Eigen::VectorXd & prev_control, int timestep, CommonBranchData & cbd, 
        //     SpecificBranchData & sbd, WallClockTimeData & wctdata) {

        //     // compute vector of vectors 
        //     Eigen::MatrixXd U = m_ground_mdp->U(); // (m , 2)
        //     std::vector<std::vector<double>> m_discrete_actions_per_dimension;
        //     for (int ii=0; ii<m_action_dim; ii++) {
        //         Eigen::VectorXd eig_linspace = Eigen::VectorXd::LinSpaced(m_num_action_discretization, U(ii,0), U(ii,1));
        //         std::vector<double> std_linspace(eig_linspace.data(), eig_linspace.data() + eig_linspace.size());
        //         m_discrete_actions_per_dimension.push_back(std_linspace);
        //     }
        //     std::vector<std::vector<double>> m_discrete_actions_temp = generateCombinations(m_discrete_actions_per_dimension);

        //     if (branch_idx > (m_discrete_actions_temp.size()-1)) {
        //         return false;
        //     }

        //     // hold ith vector control input for H timesteps
        //     Eigen::VectorXd u = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(m_discrete_actions_temp[branch_idx].data(), m_action_dim);
        //     std::vector<Eigen::VectorXd> us(m_decision_making_horizon, u);
        //     Trajectory traj = rollout_action_sequence(prev_state, us, m_ground_mdp, true);
        //     sbd.xs = traj.xs;
        //     sbd.rs = traj.rs;
        //     sbd.us = traj.us;
            
        //     return traj.is_valid;
        // }


        bool expand_ii_once(RNG& rng, int branch_idx, const Eigen::VectorXd & prev_state, 
            const Eigen::VectorXd & prev_control, int timestep, CommonBranchData & cbd, 
            SpecificBranchData & sbd, WallClockTimeData & wctdata) {

            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            wctdata.cbd_empty = cbd.empty;
            #endif

            cbd.zbar0 = prev_state.head(m_state_dim-1);

            if (m_verbose) { 
                std::cout << "" << std::endl; 
                std::cout << "ss_expansion_datalogging..." << std::endl; 
                std::cout << "branch_idx: " << branch_idx << std::endl;
                std::cout << "cbd.zbar0: " << std::endl; print_v(cbd.zbar0);
            }

            if (branch_idx > (m_num_spectral_actions+m_max_num_special_actions-1) 
                    || (branch_idx == 0 && !m_empty_action_on) 
                    || (branch_idx == 1 && !m_equ_action_on) 
                    || (branch_idx == 2 && !m_greedy_action_on) 
                    || (branch_idx == 3 && !m_greedy_action2_on)
                    || (branch_idx == 4 && !m_greedy_action3_on)) {

                # if _DOTS_HPP_PROFILING_ON_
                wctdata.breakpoints.push_back(std::chrono::system_clock::now());
                # endif
                return false;
            }

            // define state and input space 
            // Eigen::MatrixXd X = m_ground_mdp->X();
            Eigen::MatrixXd X = scale_cube(m_ground_mdp->X(), 0.95);
            Eigen::MatrixXd U = m_ground_mdp->U();
            Eigen::MatrixXd V(m_action_dim, 2);
            V.col(0) = Eigen::VectorXd::Constant(m_action_dim,  0.0);
            V.col(1) = Eigen::VectorXd::Constant(m_action_dim,  1.0);

            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            #endif

            // common computations ... this may or may not help us .. computation vs memory tradeoff 
            if (!cbd.empty && timestep == 0) {
                int a=1;
            } else {

                #if _DOTS_HPP_PROFILING_ON_
                wctdata.breakpoints.push_back(std::chrono::system_clock::now());
                #endif

                // compute input transformation 
                if (m_verbose) { std::cout << "input transformation..." << std::endl; }
                for (int ii=0; ii<m_action_dim; ii++){
                    double du = U(ii,1) - U(ii,0);
                    cbd.S(ii,ii) = 1 / du; 
                    cbd.b(ii) = -1 * U(ii,0) / du; 
                    cbd.S_inv(ii,ii) = du; 
                }
                cbd.S_inv_b = cbd.S_inv * cbd.b;
                if (m_verbose) { 
                    std::cout << "S: " << std::endl; print_m(cbd.S); 
                    std::cout << "S_inv: " << std::endl; print_m(cbd.S_inv); 
                    std::cout << "b: " << std::endl; print_v(cbd.b); 
                }

                // choice of initial control
                if (m_initialize_mode == 0) {
                    cbd.ubar0 = m_ground_mdp->empty_control();
                } else if (m_initialize_mode == 1) {
                    cbd.ubar0 = prev_control;
                } else if (m_initialize_mode == 2) {
                    throw std::logic_error("thermal compensation not implemented, check commits before 11/20/2023");
                } else if (m_initialize_mode == 3) {
                    cbd.ubar0 = m_ground_mdp->empty_control();
                }
                cbd.vbar0 = cbd.S * cbd.ubar0 + cbd.b;
                if (m_verbose) {
                    std::cout << "cbd.zbar0" << std::endl; print_v(cbd.zbar0);
                    std::cout << "cbd.ubar0" << std::endl; print_v(cbd.ubar0);
                    std::cout << "cbd.vbar0" << std::endl; print_v(cbd.vbar0);
                }

                #if _DOTS_HPP_PROFILING_ON_
                wctdata.breakpoints.push_back(std::chrono::system_clock::now());
                #endif

                // linearization trajectory: xbars, ubars, vbars, As, Bs, cs, Ks where: 
                // zbar_{k+1} = A_k z_k + B_k u_{k+1} + c_k
                if (m_verbose) { std::cout << "initialization..." << std::endl; }
                
                for (int kk=0; kk<m_dynamics_horizon; kk++) {

                    if (m_verbose) { std::cout << "approximate dynamics..." << std::endl; }
                    // approximate dynamics
                    if (kk==0) {
                        // cbd.As[kk] = m_ground_mdp->dFdx_non_augmented(cbd.zbar0, cbd.ubar0);
                        // cbd.Bs[kk] = m_ground_mdp->dFdu_non_augmented(cbd.zbar0, cbd.ubar0) * cbd.S_inv;
                        // cbd.cs[kk] = m_ground_mdp->F_non_augmented(cbd.zbar0, cbd.ubar0) 
                        //     - cbd.As[kk] * cbd.zbar0 - cbd.Bs[kk] * cbd.vbar0; 
                        cbd.As[kk] = m_ground_mdp->dFdx(prev_state, cbd.ubar0).block(0,0,m_state_dim-1,m_state_dim-1);
                        cbd.Bs[kk] = m_ground_mdp->dFdu(prev_state, cbd.ubar0).block(0,0,m_state_dim-1,m_action_dim) * cbd.S_inv;
                        cbd.cs[kk] = m_ground_mdp->F(prev_state, cbd.ubar0).head(m_state_dim-1) 
                            - cbd.As[kk] * cbd.zbar0 - cbd.Bs[kk] * cbd.vbar0; 
                    } else {
                        cbd.As[kk] = cbd.As[kk-1];
                        cbd.Bs[kk] = cbd.Bs[kk-1];
                        cbd.cs[kk] = cbd.cs[kk-1]; 
                    }

                    if (m_verbose) { std::cout << "local controller gains..." << std::endl; }
                    // compute local controller gains
                    if (m_control_mode == 1) {
                        if (kk==0) {
                            Eigen::MatrixXd Q = m_rho;
                            Eigen::MatrixXd R = cbd.S;
                            Eigen::MatrixXd M = DiscreteAlgebraicRiccatiEquation(cbd.As[kk], cbd.Bs[kk], Q, R);
                            cbd.Ks[kk] = (R + cbd.Bs[kk].transpose() * M * cbd.Bs[kk]).completeOrthogonalDecomposition().pseudoInverse() * 
                                cbd.Bs[kk].transpose() * M * cbd.As[kk]; 
                        } else {
                            cbd.Ks[kk] = cbd.Ks[kk-1];
                        }
                    } else if (m_control_mode == 2) {
                        throw std::logic_error("continuous control not implemented, check commits before 11/20/2023");
                    } 

                    // compute next control
                    if (m_verbose) { std::cout << "next_control..." << std::endl; }
                    if (m_initialize_mode == 0) {
                        cbd.ubars[kk] = cbd.ubar0;
                    } else if (m_initialize_mode == 1) {
                        cbd.ubars[kk] = prev_control;
                    } else if (m_initialize_mode == 2) {
                        throw std::logic_error("thermal compensation not implemented, check commits before 11/20/2023");
                    } else if (m_initialize_mode == 3) {
                        // if (kk == 0) {
                        //     cbd.ubars[kk] = cbd.ubar0 
                        //         - cbd.Ks[kk].block(0,3,4,3) * cbd.zbar0.segment(3,3) // set velocities to zero
                        //         - cbd.Ks[kk].block(0,6,4,2) * cbd.zbar0.segment(6,2) // set pitch and roll to zero
                        //         - cbd.Ks[kk].block(0,9,4,3) * cbd.zbar0.segment(9,3); // set angular rates to zero
                        // } else {
                        //     cbd.ubars[kk] = cbd.ubar0 
                        //         - cbd.Ks[kk].col(2) * (cbd.zbars[kk-1](2) - cbd.zbar0(2)) // maintain initial z position
                        //         - cbd.Ks[kk].block(0,3,4,3) * cbd.zbar0.segment(3,3) // set velocities to zero
                        //         - cbd.Ks[kk].block(0,6,4,2) * cbd.zbar0.segment(6,2) // set pitch and roll to zero
                        //         - cbd.Ks[kk].block(0,9,4,3) * cbd.zbar0.segment(9,3); // set angular rates to zero;
                        // }

                        if (kk == 0) {
                            cbd.ubars[kk] = cbd.ubar0 
                                - cbd.Ks[kk].block(0,6,m_action_dim,6) * cbd.zbar0.segment(6,6);
                        } else {
                            cbd.ubars[kk] = cbd.ubar0 
                                - cbd.Ks[kk].block(0,6,m_action_dim,6) * cbd.zbars[kk-1].segment(6,6) 
                                - cbd.Ks[kk].col(2) * (cbd.zbars[kk-1](2) - cbd.zbar0(2))
                                - cbd.Ks[kk].col(5) * (cbd.zbars[kk-1](5));
                        }

                        // if (kk == 0) {
                        //     cbd.ubars[kk] = cbd.ubar0 
                        //         - cbd.Ks[kk].block(0,3,4,3) * cbd.zbar0.segment(3,3)  
                        //         - cbd.Ks[kk].block(0,6,4,2) * cbd.zbar0.segment(6,2) 
                        //         - cbd.Ks[kk].block(0,9,4,3) * cbd.zbar0.segment(9,3); 
                        // } else {
                        //     cbd.ubars[kk] = cbd.ubar0 
                        //         - cbd.Ks[kk].col(2) * (cbd.zbars[kk-1](2) - cbd.zbar0(2)) // maintain initial z position
                        //         - cbd.Ks[kk].col(5) * (cbd.zbars[kk-1](2) - cbd.zbar0(5)) // maintain initial z position
                        //         - cbd.Ks[kk].block(0,3,4,3) * cbd.zbars[kk-1].segment(3,3) // set velocities to zero
                        //         - cbd.Ks[kk].block(0,6,4,2) * cbd.zbars[kk-1].segment(6,2) // set pitch and roll to zero
                        //         - cbd.Ks[kk].block(0,9,4,3) * cbd.zbars[kk-1].segment(9,3); // set angular rates to zero;
                        // }

                        clip(cbd.ubars[kk], U);
                    }

                    // compute normalized control 
                    cbd.vbars[kk] = cbd.S * cbd.ubars[kk] + cbd.b;

                    if (m_verbose) { std::cout << "next_state..." << std::endl; }
                    // compute next state
                    if (kk==0){
                        cbd.zbars[kk] = cbd.As[kk] * cbd.zbar0 + cbd.Bs[kk] * cbd.vbars[kk] + cbd.cs[kk];
                    } else {
                        cbd.zbars[kk] = cbd.As[kk] * cbd.zbars[kk-1] + cbd.Bs[kk] * cbd.vbars[kk] + cbd.cs[kk];
                    }

                    if (m_verbose) { 
                        std::cout << "kk: " << kk << std::endl; 
                        std::cout << "zbars[kk]: " << std::endl; print_v(cbd.zbars[kk]); 
                        std::cout << "ubars[kk]: " << std::endl; print_v(cbd.ubars[kk]); 
                        std::cout << "vbars[kk]: " << std::endl; print_v(cbd.vbars[kk]); 
                        std::cout << "As[kk]: " << std::endl; print_m(cbd.As[kk]); 
                        std::cout << "Bs[kk]: " << std::endl; print_m(cbd.Bs[kk]); 
                        std::cout << "cs[kk]: " << std::endl; print_v(cbd.cs[kk]); 
                        std::cout << "Ks[kk]: " << std::endl; print_m(cbd.Ks[kk]); 
                    }

                }
                #if _DOTS_HPP_PROFILING_ON_
                wctdata.breakpoints.push_back(std::chrono::system_clock::now());
                #endif

                if (m_verbose) { std::cout << "controllability..." << std::endl; }
                // compute products of "A": [I, A_{H-1}, A_{H-1} A_{H-2}, ..., (prod_{k=0}^{H-1} A_k)]
                std::vector<Eigen::MatrixXd> A_powers(m_dynamics_horizon);
                A_powers[0] = Eigen::MatrixXd::Identity(m_state_dim-1,m_state_dim-1);
                for (int kk=1; kk<m_dynamics_horizon; kk++){
                    A_powers[kk] = A_powers[kk-1] * cbd.As[m_dynamics_horizon-kk];
                }
                // compute controllability matrix: C = [(prod_{k=0}^{H-1} A_k) B_0, (prod_{k=1}^{H-1} A_k) B_1, ..., A_{H-1} B_{H-2}, B_{H-1}]
                for (int kk=0; kk<m_dynamics_horizon; kk++){
                    cbd.C.block(0,kk*m_action_dim,m_state_dim-1,m_action_dim) = A_powers[m_dynamics_horizon-kk-1] * cbd.Bs[kk]; 
                }
                if (m_verbose) { std::cout << "C: " << std::endl; print_m(cbd.C); print_m_specs(cbd.C); }

                if (m_verbose) { std::cout << "C vbars_H..." << std::endl; }
                for (int kk=0; kk<m_dynamics_horizon; kk++){
                    cbd.C_vbars_H += cbd.C.block(0, kk*m_action_dim, m_state_dim-1, m_action_dim) * cbd.vbars[kk]; 
                }
                if (m_verbose) { std::cout << "C vbars_H" << std::endl; print_v(cbd.C_vbars_H); }
                #if _DOTS_HPP_PROFILING_ON_
                wctdata.breakpoints.push_back(std::chrono::system_clock::now());
                #endif

                // spectrum and pinv
                if (m_verbose) { std::cout << "spectrum..." << std::endl; }
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(cbd.C * cbd.C.transpose());
                cbd.eigenVectors = eigen_solver.eigenvectors().rowwise().reverse(); 
                cbd.eigenValues = eigen_solver.eigenvalues().reverse(); 

                if (m_verbose) { std::cout << "pinv..." << std::endl; }
                // cbd.C_pinv = cbd.C.completeOrthogonalDecomposition().pseudoInverse(); // redo computation 
                Eigen::VectorXd eigenValuesInv = Eigen::VectorXd::Zero(m_state_dim-1);
                for (int ii=0; ii<m_state_dim-1; ii++){
                    if (cbd.eigenValues(ii) > 1e-12) { eigenValuesInv(ii) = 1.0 / cbd.eigenValues(ii); } 
                }
                cbd.C_pinv = cbd.C.transpose() * cbd.eigenVectors * eigenValuesInv.asDiagonal() * cbd.eigenVectors.transpose();
                cbd.eigenValuesSqrt = cbd.eigenValues.cwiseSqrt();
                if (m_verbose) { 
                    std::cout << "eigenVectors: " << std::endl; print_m(cbd.eigenVectors); 
                    std::cout << "eigenValues: " << std::endl; print_v(cbd.eigenValues); 
                    std::cout << "eigenValuesSqrt: " << std::endl; print_v(cbd.eigenValuesSqrt); 
                    std::cout << "C_pinv: " << std::endl; print_m_specs(cbd.C_pinv); 
                }
                #if _DOTS_HPP_PROFILING_ON_
                wctdata.breakpoints.push_back(std::chrono::system_clock::now());
                #endif

                // find the relevant eigenvectors 
                std::vector<int> idxs_to_maximize;
                if (m_spectral_branches_mode == 0) {
                    throw std::logic_error("m_spectral_branches_mode == 0 not implemented");
                } else if (m_spectral_branches_mode == 1) {
                    idxs_to_maximize = m_ground_mdp->velocity_idxs();
                } else if (m_spectral_branches_mode == 2) {
                    idxs_to_maximize = m_ground_mdp->position_idxs();
                } else if (m_spectral_branches_mode == 3) {
                    idxs_to_maximize = m_ground_mdp->my_idxs();
                }
                cbd.eigenVectors_to_search.resize(m_state_dim-1, idxs_to_maximize.size());
                cbd.eigenValues_to_search.resize(idxs_to_maximize.size());
                std::vector<int> idxs_to_search; 
                for (int ii=0; ii<idxs_to_maximize.size(); ii++){
                    int best_idx = 0; 
                    double best_value = 0.0;
                    for (int jj=0; jj<m_state_dim-1; jj++){
                        double curr_value = cbd.eigenValues(idxs_to_maximize[ii]) * std::abs(cbd.eigenVectors(idxs_to_maximize[ii], jj));
                        if (curr_value > best_value && !is_int_in_int_vec(jj, idxs_to_search)) {
                            best_value = curr_value; 
                            best_idx = jj;
                        }
                    }
                    idxs_to_search.push_back(best_idx);
                }
                for (int ii=0; ii<idxs_to_search.size(); ii++){
                    cbd.eigenVectors_to_search.col(ii) = cbd.eigenVectors.col(idxs_to_search[ii]);
                    cbd.eigenValues_to_search(ii) = cbd.eigenValues(idxs_to_search[ii]);
                }
                cbd.eigenValuesSqrt_to_search = cbd.eigenValues_to_search.cwiseSqrt();

                // dont compute again 
                cbd.empty = false;
            }
            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            #endif

            // specific compute 
            if (m_verbose) { std::cout << "delta_z_H_unscaled..." << std::endl; }
            if (branch_idx == 0 && m_empty_action_on) { ;
                // do nothing, delta_z is zero, so control input will be empty 
            } else if (branch_idx == 1 && m_equ_action_on) {
                Eigen::VectorXd z_H_des = Eigen::VectorXd::Zero(m_state_dim-1); 
                z_H_des(0) = cbd.zbars[m_dynamics_horizon-1](0); // set x positions to zbar_H
                z_H_des(1) = cbd.zbars[m_dynamics_horizon-1](1); // set y positions to zbar_H
                z_H_des(2) = cbd.zbar0(2); // set z positions to zbar0
                z_H_des.segment(3,3) = Eigen::VectorXd::Zero(3); // set velocities to zero 
                z_H_des(6) = 0; // set roll to zero
                z_H_des(7) = 0; // set pitch to zero
                z_H_des(8) = cbd.zbars[m_dynamics_horizon-1](8); // set pitch to zbar_H
                z_H_des.segment(9,3) = Eigen::VectorXd::Zero(3); // set angular velocities to zero 
                sbd.delta_z_H_unscaled = project_vector_onto_ellipse(z_H_des - cbd.zbars[m_dynamics_horizon-1], cbd.eigenVectors, m_greedy_gain*cbd.eigenValuesSqrt);

            } else if (branch_idx == 2 && m_greedy_action_on) {
                // Eigen::VectorXd des_velocity_wf = m_ground_mdp->get_xd().head(3) - cbd.zbars[m_dynamics_horizon-1].head(3);
                // double des_yaw = std::atan2(des_velocity_wf(1), des_velocity_wf(0));
                // Eigen::VectorXd des_velocity_bf = rot_mat_body_to_inertial(cbd.zbars[m_dynamics_horizon-1](6), 
                //     cbd.zbars[m_dynamics_horizon-1](7), cbd.zbars[m_dynamics_horizon-1](8)).transpose() * des_velocity_wf;
                // Eigen::VectorXd delta_z_H_preproj = Eigen::VectorXd::Zero(m_state_dim-1);
                // double k = (1 - m_greedy_rate) / (m_dynamics_horizon * m_ground_mdp->dt());
                // if (des_velocity_wf.norm() > m_greedy_min_dist) {
                //     delta_z_H_preproj.segment(3,3) = k * des_velocity_bf - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                // } else {
                //     delta_z_H_preproj.segment(3,3) = - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                // }
                // delta_z_H_preproj.segment(6,6) = -cbd.zbars[m_dynamics_horizon-1].segment(6,6);
                // delta_z_H_preproj(11) = k * des_yaw - cbd.zbars[m_dynamics_horizon-1](11);
                // sbd.delta_z_H_unscaled = project_vector_onto_ellipse(delta_z_H_preproj, cbd.eigenVectors, m_greedy_gain*cbd.eigenValuesSqrt);

                // Eigen::VectorXd x_des = m_ground_mdp->get_xd().segment(0, 12);
                // Eigen::VectorXd des_velocity_wf = m_ground_mdp->get_xd().head(3) - cbd.zbars[m_dynamics_horizon-1].head(3);
                // Eigen::VectorXd des_velocity_bf = rot_mat_body_to_inertial(cbd.zbars[m_dynamics_horizon-1](6), 
                //     cbd.zbars[m_dynamics_horizon-1](7), cbd.zbars[m_dynamics_horizon-1](8)).transpose() * des_velocity_wf;
                // Eigen::VectorXd delta_z_H_preproj = Eigen::VectorXd::Zero(m_state_dim-1);
                // double k = (1 - m_greedy_rate) / (m_dynamics_horizon * m_ground_mdp->dt());
                // delta_z_H_preproj.segment(3,3) = k * des_velocity_bf - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                // delta_z_H_preproj.segment(6,6) = -cbd.zbars[m_dynamics_horizon-1].segment(6,6);
                // sbd.delta_z_H_unscaled = project_vector_onto_ellipse(delta_z_H_preproj, cbd.eigenVectors, m_greedy_gain*cbd.eigenValuesSqrt);

                // find closest target 
                Eigen::VectorXd best_target = Eigen::VectorXd::Zero(13); 
                double best_target_dist = 1000.0;
                int num_targets = m_ground_mdp->get_xd().rows() / 13; 
                for (int ii=0; ii<num_targets; ii++) {
                    Eigen::VectorXd target = m_ground_mdp->get_xd().segment(ii*13,13);
                    double target_dist = (target.head(3) - cbd.zbars[m_dynamics_horizon-1].head(3)).norm();
                    if (target_dist < best_target_dist) {
                        best_target_dist = target_dist;
                        best_target = target;
                    }
                }

                // track it 
                Eigen::VectorXd des_velocity_wf = best_target.head(3) - cbd.zbars[m_dynamics_horizon-1].head(3);
                Eigen::VectorXd des_velocity_bf = rot_mat_body_to_inertial(cbd.zbars[m_dynamics_horizon-1](6), 
                    cbd.zbars[m_dynamics_horizon-1](7), cbd.zbars[m_dynamics_horizon-1](8)).transpose() * des_velocity_wf;
                Eigen::VectorXd delta_z_H_preproj = Eigen::VectorXd::Zero(m_state_dim-1);
                double k = (1 - m_greedy_rate) / (m_dynamics_horizon * m_ground_mdp->dt());
                delta_z_H_preproj.segment(3,3) = k * des_velocity_bf - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                delta_z_H_preproj.segment(6,6) = -cbd.zbars[m_dynamics_horizon-1].segment(6,6);
                sbd.delta_z_H_unscaled = project_vector_onto_ellipse(delta_z_H_preproj, cbd.eigenVectors, m_greedy_gain*cbd.eigenValuesSqrt);

            } else if (branch_idx == 3 && m_greedy_action2_on){
                // Eigen::VectorXd x_des = m_ground_mdp->get_xd().tail(m_state_dim).head(m_state_dim-1);
                // Eigen::VectorXd des_velocity_wf = x_des.head(3) - cbd.zbars[m_dynamics_horizon-1].head(3);
                // double des_yaw = std::atan2(des_velocity_wf(1), des_velocity_wf(0));
                // Eigen::VectorXd des_velocity_bf = rot_mat_body_to_inertial(cbd.zbars[m_dynamics_horizon-1](6), 
                //     cbd.zbars[m_dynamics_horizon-1](7), cbd.zbars[m_dynamics_horizon-1](8)).transpose() * des_velocity_wf;
                // Eigen::VectorXd delta_z_H_preproj = Eigen::VectorXd::Zero(m_state_dim-1);
                // double k = (1 - m_greedy_rate) / (m_dynamics_horizon * m_ground_mdp->dt());
                // if (des_velocity_wf.norm() > m_greedy_min_dist) {
                //     delta_z_H_preproj.segment(3,3) = k * des_velocity_bf - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                // } else {
                //     delta_z_H_preproj.segment(3,3) = - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                // }
                // delta_z_H_preproj.segment(3,3) = k * des_velocity_bf - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                // delta_z_H_preproj.segment(6,6) = -cbd.zbars[m_dynamics_horizon-1].segment(6,6);
                // delta_z_H_preproj(11) = k * des_yaw - cbd.zbars[m_dynamics_horizon-1](11);
                // sbd.delta_z_H_unscaled = project_vector_onto_ellipse(delta_z_H_preproj, cbd.eigenVectors, m_greedy_gain*cbd.eigenValuesSqrt);

                // Eigen::VectorXd x_des = m_ground_mdp->get_xd().tail(m_state_dim).head(m_state_dim-1);
                Eigen::VectorXd x_des = m_ground_mdp->get_xd().segment(13, 12);
                Eigen::VectorXd des_velocity_wf = x_des.head(3) - cbd.zbars[m_dynamics_horizon-1].head(3);
                Eigen::VectorXd des_velocity_bf = rot_mat_body_to_inertial(cbd.zbars[m_dynamics_horizon-1](6), 
                    cbd.zbars[m_dynamics_horizon-1](7), cbd.zbars[m_dynamics_horizon-1](8)).transpose() * des_velocity_wf;
                Eigen::VectorXd delta_z_H_preproj = Eigen::VectorXd::Zero(m_state_dim-1);
                double k = (1 - m_greedy_rate) / (m_dynamics_horizon * m_ground_mdp->dt());
                delta_z_H_preproj.segment(3,3) = k * des_velocity_bf - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                delta_z_H_preproj.segment(6,6) = -cbd.zbars[m_dynamics_horizon-1].segment(6,6);
                sbd.delta_z_H_unscaled = project_vector_onto_ellipse(delta_z_H_preproj, cbd.eigenVectors, m_greedy_gain*cbd.eigenValuesSqrt);

            } else if (branch_idx == 4 && m_greedy_action3_on){
                Eigen::VectorXd x_des = m_ground_mdp->get_xd().segment(26, 12);
                Eigen::VectorXd des_velocity_wf = x_des.head(3) - cbd.zbars[m_dynamics_horizon-1].head(3);
                Eigen::VectorXd des_velocity_bf = rot_mat_body_to_inertial(cbd.zbars[m_dynamics_horizon-1](6), 
                    cbd.zbars[m_dynamics_horizon-1](7), cbd.zbars[m_dynamics_horizon-1](8)).transpose() * des_velocity_wf;
                Eigen::VectorXd delta_z_H_preproj = Eigen::VectorXd::Zero(m_state_dim-1);
                double k = (1 - m_greedy_rate) / (m_dynamics_horizon * m_ground_mdp->dt());
                delta_z_H_preproj.segment(3,3) = k * des_velocity_bf - cbd.zbars[m_dynamics_horizon-1].segment(3,3);
                delta_z_H_preproj.segment(6,6) = -cbd.zbars[m_dynamics_horizon-1].segment(6,6);
                sbd.delta_z_H_unscaled = project_vector_onto_ellipse(delta_z_H_preproj, cbd.eigenVectors, m_greedy_gain*cbd.eigenValuesSqrt);
            } else {
                int eigen_idx = (branch_idx-m_max_num_special_actions)/2; // integer division 
                int sign = -2 * int((branch_idx-m_max_num_special_actions) % 2) + 1;
                if (m_verbose) { std::cout << "eigen_idx: " << eigen_idx << std::endl; std::cout << "sign: " << sign << std::endl; }
                sbd.delta_z_H_unscaled = sign * cbd.eigenValuesSqrt_to_search(eigen_idx) * cbd.eigenVectors_to_search.col(eigen_idx);
                // damping
                if (m_verbose) { std::cout << "damping..." << std::endl; }
                if (m_damping_mode == 0) {
                    // damp wrt absolute terminal state 
                    Eigen::VectorXd z_H = sbd.delta_z_H_unscaled + cbd.zbars[m_dynamics_horizon-1]; 
                    for (int ii=0; ii<m_state_dim-1; ii++){
                        z_H(ii) = (1.0-m_damping_gains[ii]) * z_H(ii);
                    }
                    sbd.delta_z_H_unscaled = z_H - cbd.zbars[m_dynamics_horizon-1];  
                } else if (m_damping_mode == 1) {
                    // damp wrt displacement
                    for (int ii=0; ii<m_state_dim-1; ii++){
                        sbd.delta_z_H_unscaled(ii) = (1.0-m_damping_gains[ii]) * sbd.delta_z_H_unscaled(ii);
                    }
                } else if (m_damping_mode == 2) {
                    // damp wrt initial state
                    Eigen::VectorXd z_H = sbd.delta_z_H_unscaled + cbd.zbars[m_dynamics_horizon-1]; 
                    Eigen::VectorXd desired_zH_minus_x0 = z_H - cbd.zbars[0];
                    for (int ii=0; ii<m_state_dim-1; ii++){
                        desired_zH_minus_x0(ii) = (1.0-m_damping_gains[ii]) * desired_zH_minus_x0(ii);
                    }
                    Eigen::VectorXd desired_z_H = desired_zH_minus_x0 + cbd.zbars[0];
                    sbd.delta_z_H_unscaled = desired_z_H - cbd.zbars[m_dynamics_horizon-1];
                }
            }
            if (m_verbose) { std::cout << "delta_z_H_unscaled: " << std::endl; print_v(sbd.delta_z_H_unscaled); }
            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            #endif


            // state clip
            if (m_verbose) { std::cout << "delta_z_H..." << std::endl; }
            if (m_scale_mode == 0) {
                sbd.delta_z_H = sbd.delta_z_H_unscaled;
            } else if (m_scale_mode == 1) {
                // "absolute" clipping
                Eigen::VectorXd z_H = sbd.delta_z_H_unscaled + cbd.zbars[m_dynamics_horizon-1]; 
                clip(z_H, X);
                sbd.delta_z_H = z_H - cbd.zbars[m_dynamics_horizon-1]; 
            } else if (m_scale_mode == 2 || m_scale_mode == 3) {
                // "displacement" clipping
                Eigen::VectorXd z_H = sbd.delta_z_H_unscaled + cbd.zbars[m_dynamics_horizon-1]; 
                double eta = 1.0;
                // double eta = 0.9;
                for (int ii=0; ii<m_state_dim-1; ii++){
                    if (z_H(ii) > X(ii,1)) {
                        // if following condition is not true, rollout will probably fail, but that is desired behavior
                        if (cbd.zbars[m_dynamics_horizon-1](ii) < X(ii,1)) { 
                            eta = std::min(eta, (X(ii,1) - cbd.zbars[m_dynamics_horizon-1](ii)) / (sbd.delta_z_H_unscaled(ii) + 1e-6));
                        } 
                    } else if (z_H(ii) < X(ii,0)) {
                        if (cbd.zbars[m_dynamics_horizon-1](ii) > X(ii,0)) {
                            eta = std::min(eta, (X(ii,0) - cbd.zbars[m_dynamics_horizon-1](ii)) / (sbd.delta_z_H_unscaled(ii) - 1e-6));
                        } 
                    }
                }
                sbd.delta_z_H = eta * sbd.delta_z_H_unscaled;
            }
            if (m_verbose) { std::cout << "delta_z_H: " << std::endl; print_v(sbd.delta_z_H); }
            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            #endif


            // compute normalized, clipped, control sequence 
            if (m_verbose) { std::cout << "compute vs_ref..." << std::endl; }
            for (int kk=0; kk<m_dynamics_horizon; kk++){
                sbd.vs_ref[kk] = cbd.C_pinv.block(kk*m_action_dim, 0, m_action_dim, m_state_dim-1) * (sbd.delta_z_H + cbd.C_vbars_H);
            }
            if (m_verbose) { std::cout << "vs_ref: " << std::endl; print_vv(sbd.vs_ref); }


            if (m_verbose) { std::cout << "input scaling..." << std::endl; }
            if (m_scale_mode == 0) { ;
            } else if (m_scale_mode == 1) {
                // "absolute" clipping
                for (int kk=0; kk<m_dynamics_horizon; kk++){ clip(sbd.vs_ref[kk], V); }
            } else if (m_scale_mode == 2) {
                // "displacement" clipping
                for (int kk=0; kk<m_dynamics_horizon; kk++){
                    double eta = 1.0;
                    Eigen::VectorXd delta_v_kk = sbd.vs_ref[kk] - cbd.vbars[kk]; 
                    for (int ii=0; ii<m_action_dim; ii++){
                        if (sbd.vs_ref[kk](ii) > V(ii,1)) {
                            eta = std::min(eta, (V(ii,1) - cbd.vbars[kk](ii)) / (delta_v_kk(ii) + 1e-6));
                        } else if (sbd.vs_ref[kk](ii) < V(ii,0)) {
                            eta = std::min(eta, (V(ii,0) - cbd.vbars[kk](ii)) / (delta_v_kk(ii) - 1e-6));
                        }
                    }
                    sbd.vs_ref[kk] = eta * delta_v_kk + cbd.vbars[kk];
                }
            } else if (m_scale_mode == 3) {
                // "displacement" clipping
                std::vector<double> etas(m_dynamics_horizon, 1.0);
                for (int kk=0; kk<m_dynamics_horizon; kk++){
                    double eta = 1.0;
                    Eigen::VectorXd delta_v_kk = sbd.vs_ref[kk] - cbd.vbars[kk]; 
                    for (int ii=0; ii<m_action_dim; ii++){
                        if (sbd.vs_ref[kk](ii) > V(ii,1)) {
                            eta = std::min(eta, (V(ii,1) - cbd.vbars[kk](ii)) / (delta_v_kk(ii) + 1e-6));
                        } else if (sbd.vs_ref[kk](ii) < V(ii,0)) {
                            eta = std::min(eta, (V(ii,0) - cbd.vbars[kk](ii)) / (delta_v_kk(ii) - 1e-6));
                        }
                    }
                    etas[kk] = eta;
                }
                double eta = *std::min_element(etas.begin(), etas.end()); 
                for (int kk=0; kk<m_dynamics_horizon; kk++){
                    Eigen::VectorXd delta_v_kk = sbd.vs_ref[kk] - cbd.vbars[kk]; 
                    sbd.vs_ref[kk] = eta * delta_v_kk + cbd.vbars[kk];
                }
            }
            if (m_verbose) { std::cout << "(scaled) vs_ref: " << std::endl; print_vv(sbd.vs_ref); }


            // clipped unnormalized control sequence 
            if (m_verbose) { std::cout << "compute us_ref..." << std::endl; }
            for (int kk=0; kk<m_dynamics_horizon; kk++){
                sbd.us_ref[kk] = cbd.S_inv * sbd.vs_ref[kk] - cbd.S_inv_b;
            }
            if (m_verbose) { std::cout << "us_ref: " << std::endl; print_vv(sbd.us_ref); }
            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            #endif


            // rollout zs (only used in control mode on)
            if (m_verbose) { std::cout << "compute zs_ref..." << std::endl; }
            sbd.zs_ref[0] = cbd.As[0] * cbd.zbars[0] + cbd.Bs[0] * sbd.vs_ref[0] + cbd.cs[0]; 
            for (int kk=1; kk<m_dynamics_horizon; kk++){
                sbd.zs_ref[kk] = cbd.As[kk] * sbd.zs_ref[kk-1] + cbd.Bs[kk] * sbd.vs_ref[kk] + cbd.cs[kk]; 
            }
            if (m_verbose) { std::cout << "zs_ref: " << std::endl; print_vv(sbd.zs_ref); }

            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            #endif

            // rollout xs
            if (m_verbose) { std::cout << "compute xs, rs..." << std::endl; }
            bool is_valid = true;
            sbd.us[0] = sbd.us_ref[0];
            clip(sbd.us[0], U);
            sbd.xs[0] = m_ground_mdp->F(prev_state, sbd.us[0]);
            sbd.rs[0] = m_ground_mdp->R(prev_state, sbd.us[0]);
            for (int kk=1; kk<m_dynamics_horizon; kk++){
                // open loop linear system inputs 
                if (m_control_mode == 0) {
                    sbd.us[kk] = sbd.us_ref[kk];
                // closed loop linear system inputs 
                } else if (m_control_mode == 1 || m_control_mode == 2) {
                    sbd.us[kk] = sbd.us_ref[kk] - cbd.Ks[kk] * (sbd.xs[kk-1].head(m_state_dim-1) - sbd.zs_ref[kk-1]);
                } 
                clip(sbd.us[kk], U);
                sbd.xs[kk] = m_ground_mdp->F(sbd.xs[kk-1], sbd.us[kk]);
                sbd.rs[kk] = m_ground_mdp->R(sbd.xs[kk], sbd.us[kk]);
                if (!m_ground_mdp->is_state_valid(sbd.xs[kk])) {
                    is_valid = false;
                    sbd.xs.resize(kk+1);
                    sbd.us.resize(kk+1);
                    sbd.rs.resize(kk+1);
                    break;
                }
            }
            if (m_verbose) { std::cout << "xs: " << std::endl; print_vv(sbd.xs); }
            if (m_verbose) { std::cout << "us: " << std::endl; print_vv(sbd.us); }
            if (m_verbose) { std::cout << "rs: " << std::endl; print_v(sbd.rs); }

            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());
            #endif

            if (m_verbose) { std::cout << "is_valid: " << is_valid << std::endl; }
            if (m_verbose) { std::cout << "expansion complete!" << std::endl; }
            #if _DOTS_HPP_PROFILING_ON_
            wctdata.breakpoints.push_back(std::chrono::system_clock::now());

            std::vector<std::string> labels;
            if (wctdata.cbd_empty) {
                labels = { "define_state_input_space", "cbd_initialization", "input_transformation", "linearization", 
                    "compute_controllability_matrix", "compute_spectrum", "finish_cbd", 
                    "compute_delta_z_H_unscaled", "compute_delta_z_H", "compute_vsus_ref", "compute_zs_ref", "compute_xsusrs", "finish_sbd", "total" };
            } else {
                labels = { "define_state_input_space", "finish_cbd", 
                    "compute_delta_z_H_unscaled", "compute_delta_z_H", 
                    "compute_vsus_ref", "compute_zs_ref", "compute_xsusrs", "finish_sbd", "total" };
            }

            std::cout << "----- profiling: -----" << std::endl;
            for (int ii=0; ii<labels.size()-1; ii++){
                std::cout << labels[ii] << ": " << std::chrono::duration_cast<std::chrono::microseconds>(wctdata.breakpoints[ii+1] - wctdata.breakpoints[ii]).count() << std::endl;
            }
            std::cout << "total: " << std::chrono::duration_cast<std::chrono::microseconds>(wctdata.breakpoints[wctdata.breakpoints.size()-1] - wctdata.breakpoints[0]).count() << std::endl;

            // std::cout << "  debug: num_labels = " << labels.size() << std::endl;
            // std::cout << "  debug: num_breakpoints = " << wctdata.breakpoints.size() << std::endl;

            #endif
            return is_valid; 
        }

        void clip(Eigen::VectorXd & vec, const Eigen::MatrixXd cube) {
            for (int ii=0; ii<vec.rows(); ii++) {
                vec(ii) = std::max(std::min(vec(ii), cube(ii,1)), cube(ii,0));
            }
        }

        Eigen::MatrixXd rot_mat_body_to_inertial(double phi, double theta, double psi) {
            Eigen::MatrixXd R(3,3);
            R << cos(theta) * cos(psi), 
                sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
                cos(theta) * sin(psi), 
                sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
                cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
                -sin(theta), 
                sin(phi) * cos(theta), 
                cos(phi) * cos(theta);
            return R; 
        }

        int num_actions() override { return std::min(m_max_num_branches, m_num_spectral_actions+m_max_num_special_actions); }
        // int num_actions() override { return m_num_branches; }

        // same as ground mdp 
        int state_dim() override { return m_ground_mdp->state_dim(); }

        int action_dim() override { return m_ground_mdp->action_dim(); }

        int timestep_idx() override { return m_ground_mdp->timestep_idx(); }

        Eigen::VectorXd initial_state() override { return m_ground_mdp->initial_state(); }

        Eigen::VectorXd empty_control() override { return m_ground_mdp->empty_control(); }

        void clear_obstacles() override { return m_ground_mdp->clear_obstacles(); }

        void add_obstacle(Eigen::Matrix<double,-1,2> obstacle) override { return m_ground_mdp->add_obstacle(obstacle); }

        void clear_targets() override { return m_ground_mdp->clear_targets(); }

        void add_target(Eigen::VectorXd target) override { return m_ground_mdp->add_target(target); }

        void clear_thermals() override { return m_ground_mdp->clear_thermals(); }

        void add_thermal(Eigen::MatrixXd X_thermal, Eigen::VectorXd V_thermal) override { return m_ground_mdp->add_thermal(X_thermal, V_thermal); }

        bool is_state_valid(const Eigen::VectorXd & state) override { 
            return m_ground_mdp->is_state_valid(state); }

        Eigen::Matrix<double,-1,2> X() override { return m_ground_mdp->X(); }

        Eigen::Matrix<double,-1,2> U() override { return m_ground_mdp->U(); }

        Eigen::VectorXd F(const Eigen::VectorXd & state, 
                          const Eigen::VectorXd & action) override {
            return m_ground_mdp->F(state, action); }

        Eigen::VectorXd update_augmented_state_only(const Eigen::VectorXd & state, const Eigen::VectorXd & action) override { 
            return m_ground_mdp->update_augmented_state_only(state, action); 
        }

        double R(const Eigen::VectorXd & state, 
                const Eigen::VectorXd & action) override { 
            return m_ground_mdp->R(state, action); }

        double V(Eigen::VectorXd state) override {
            return m_ground_mdp->V(state); }

        double V(Eigen::VectorXd state, RNG& rng) override {
            return m_ground_mdp->V(state, rng); }

        int H() override { return m_ground_mdp->H(); }

        double gamma() override { return m_ground_mdp->gamma(); }

        
    private: 
        MDP* m_ground_mdp;
        int m_expansion_mode; 
        int m_spectral_branches_mode; 
        int m_initialize_mode; 
        int m_control_mode;
        int m_scale_mode;
        int m_damping_mode;
        int m_state_dim; 
        int m_action_dim;
        int m_decision_making_horizon;
        int m_dynamics_horizon;
        int m_max_num_branches; // max num actions ... set to be large and then let fail_expansion handle the rest
        int m_num_spectral_actions; 
        int m_max_num_special_actions; // used to calculate eigen_idx 
        bool m_verbose; 
        bool m_empty_action_on; 
        bool m_greedy_action_on; 
        bool m_greedy_action2_on; 
        bool m_greedy_action3_on;
        bool m_equ_action_on; 
        double m_greedy_gain; 
        double m_greedy_min_dist; 
        double m_greedy_rate; 
        // double m_rho;
        Eigen::MatrixXd m_rho;
        std::vector<double> m_damping_gains; 
        int m_num_action_discretization;
};

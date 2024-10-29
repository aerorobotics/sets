#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "mdp.hpp"
#include "../util/util.hpp"
// #include "../learning/feedforward.hpp"


class SixDOFAircraft : public MDP {

    public:

        void print_aero_coeffs(const std::array<double,6> & aero_coeffs) {
            std::cout << "C_D: " << aero_coeffs[0] << std::endl;
            std::cout << "C_L: " << aero_coeffs[1] << std::endl;
            std::cout << "C_M: " << aero_coeffs[2] << std::endl;
            std::cout << "C_Y: " << aero_coeffs[3] << std::endl;
            std::cout << "C_l: " << aero_coeffs[4] << std::endl;
            std::cout << "C_n: " << aero_coeffs[5] << std::endl; }


        std::string name() override {
            return "SixDOFAircraft";
        }


        std::vector<int> velocity_idxs() override {
            // x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r, t]
            std::vector<int> idxs{ 3, 4, 5, 9, 10, 11 };
            return idxs;
        }


        std::vector<int> position_idxs() override {
            // x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r, t]
            std::vector<int> idxs{ 0, 1, 2, 6, 7, 8 };
            return idxs;
        }

        std::vector<int> my_idxs() override {
            return m_my_idxs;
        }


        SixDOFAircraft(std::string mdp_config_path) {
            YAML::Node config = YAML::LoadFile(mdp_config_path);

            m_verbose = config["ground_mdp_verbose"].as<bool>();
            if (m_verbose) { std::cout << "setting system params" <<  std::endl;}

            // modes
            std::string flight_mode = config["flight_mode"].as<std::string>();
            if (flight_mode == "fixed_wing") { // u = [delta_e, delta_r, delta_a]
                m_flight_mode = 0;
            } else if (flight_mode == "quadrotor") { // u = [thrust_z, tau_x, tau_y, tau_z]
                m_flight_mode = 1;
            } else if (flight_mode == "transition") { // u = [delta_e, delta_r, delta_a, thrust_z, tau_x, tau_y, tau_z, thrust_x]
                m_flight_mode = 2;
            } else { 
                throw std::logic_error("flight mode not recognized");
            }

            std::string wind_mode = config["wind_mode"].as<std::string>();
            if (wind_mode == "empty") {
                m_wind_mode = 0;
            } else if (wind_mode == "thermal") {
                m_wind_mode = 1;
            } else if (wind_mode == "analytical_thermal") {
                m_wind_mode = 2;
            } else {
                throw std::logic_error("wind mode not recognized");
            }

            std::string aero_mode = config["aero_mode"].as<std::string>();
            if (aero_mode == "empty") {
                m_aero_mode = 0;
            } else if (aero_mode == "linear") {
                m_aero_mode = 1;
            } else if (aero_mode == "nonlinear") {
                m_aero_mode = 2;
            } else if (aero_mode == "neural") {
                m_aero_mode = 3;
            } else if (aero_mode == "neural_thermal") {
                m_aero_mode = 4;
            } else if (aero_mode == "neural_thermal_moment") {
                m_aero_mode = 5;
            } else if (aero_mode == "linear_with_thermal") {
                m_aero_mode = 6;
            } else {
                throw std::logic_error("aero mode not recognized");
            }

            std::string reward_mode = config["reward_mode"].as<std::string>();
            if (reward_mode == "regulation") {
                m_reward_mode = 0;
            } else if (reward_mode == "experiment") {
                m_reward_mode = 1;
            } else if (reward_mode == "observation") {
                m_reward_mode = 2;
            } else {
                throw std::logic_error("reward mode not recognized");
            }

            if (m_verbose) { std::cout << "setting system modes ok" <<  std::endl;}

            // x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r, t] 
            // positions are in global frame 
            // velocities are in body frame 
            m_state_dim = 13;

            // u = [delta_e, delta_r, delta_a, thrust_z, tau_x, tau_y, tau_z, thrust_x]
            m_action_dim = 8; 

            // system specific parameters
            m_dt = config["ground_mdp_dt"].as<double>();
            m_gamma = config["ground_mdp_gamma"].as<double>();
            m_H = config["ground_mdp_H"].as<int>();
            m_mass = config["mass"].as<double>();
            m_gravity = config["gravity"].as<double>();
            m_Ixx = config["Ixx"].as<double>();
            m_Ixz = config["Ixz"].as<double>();
            m_Iyy = config["Iyy"].as<double>();
            m_Izx = config["Izx"].as<double>();
            m_Izz = config["Izz"].as<double>();
            m_control_hold = config["ground_mdp_control_hold"].as<int>();
            m_V_alive = config["ground_mdp_V_alive"].as<double>();

            m_wind_period = config["wind_period"].as<int>();
            m_wind_duty_cycle = config["wind_duty_cycle"].as<double>();
            
            if (m_verbose) { std::cout << "setting system specific param ok" <<  std::endl;}

            if (m_reward_mode == 2) {
                m_obs_cone_angle = config["obs_cone_angle"].as<double>();
                m_obs_cone_length = config["obs_cone_length"].as<double>();
                m_obs_min_speed = config["obs_min_speed"].as<double>();
            }
            if (m_verbose) { std::cout << "setting observation model param ok" <<  std::endl;}

            m_neural_thermal_scale = config["neural_thermal_scale"].as<double>();

            if (config["my_idxs"]) {
                m_my_idxs = config["my_idxs"].as<std::vector<int>>();
            } else {
                m_my_idxs = std::vector<int>();
                assert(config["dots_spectral_branches_mode"].as<std::string>() != "my_idxs");
            }

            if (m_aero_mode == 1 || m_aero_mode == 2 || m_aero_mode == 6) {
                m_rho = config["rho"].as<double>();
                m_S = config["S"].as<double>();
                m_b = config["b"].as<double>();
                m_c = config["c"].as<double>();
    
                // takeoff 
                m_alpha_takeoff = config["alpha_takeoff"].as<double>();
                m_beta_takeoff = config["beta_takeoff"].as<double>();
                m_speed_takeoff = config["speed_takeoff"].as<double>();

                // 
                m_max_alpha = config["max_alpha"].as<double>();
                m_max_beta = config["max_beta"].as<double>();

                if (m_verbose) { std::cout << "setting takeoff conditions ok" <<  std::endl;}

                // aero model 
                // nonlinear lift / drag
                m_alpha_0 = config["alpha_0"].as<double>();
                m_M = config["M"].as<double>();
                m_C_D_p = config["C_D_p"].as<double>();
                m_oswald_eff = config["oswald_eff"].as<double>();

                if (m_verbose) { std::cout << "setting nonlinear aero model ok" <<  std::endl;}

                // linear 
                m_C_D_0 = config["C_D_0"].as<double>();
                m_C_D_alpha = config["C_D_alpha"].as<double>();
                m_C_D_q = config["C_D_q"].as<double>();
                m_C_D_delta_e = config["C_D_delta_e"].as<double>();
                m_C_L_0 = config["C_L_0"].as<double>();
                m_C_L_alpha = config["C_L_alpha"].as<double>();
                m_C_L_q = config["C_L_q"].as<double>();
                m_C_L_delta_e = config["C_L_delta_e"].as<double>();
                m_C_M_0 = config["C_M_0"].as<double>();
                m_C_M_alpha = config["C_M_alpha"].as<double>();
                m_C_M_q = config["C_M_q"].as<double>();
                m_C_M_delta_e = config["C_M_delta_e"].as<double>();
                m_C_Y_0 = config["C_Y_0"].as<double>();
                m_C_Y_beta = config["C_Y_beta"].as<double>();
                m_C_Y_p = config["C_Y_p"].as<double>();
                m_C_Y_r = config["C_Y_r"].as<double>();
                m_C_Y_delta_a = config["C_Y_delta_a"].as<double>();
                m_C_Y_delta_r = config["C_Y_delta_r"].as<double>();
                m_C_l_0 = config["C_l_0"].as<double>();
                m_C_l_beta = config["C_l_beta"].as<double>();
                m_C_l_p = config["C_l_p"].as<double>();
                m_C_l_r = config["C_l_r"].as<double>();
                m_C_l_delta_a = config["C_l_delta_a"].as<double>();
                m_C_l_delta_r = config["C_l_delta_r"].as<double>();
                m_C_n_0 = config["C_n_0"].as<double>();
                m_C_n_beta = config["C_n_beta"].as<double>();
                m_C_n_p = config["C_n_p"].as<double>();
                m_C_n_r = config["C_n_r"].as<double>();
                m_C_n_delta_a = config["C_n_delta_a"].as<double>();
                m_C_n_delta_r = config["C_n_delta_r"].as<double>();
                if (m_verbose) { std::cout << "setting aero model ok" <<  std::endl;}
            }

            std::vector<double> aero_clip_yml = config["aero_clip"].as<std::vector<double>>();
            m_aero_clip = Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(aero_clip_yml.data(), 6, 2);
            m_aero_clip.resize(6,2);
            std::vector<double> aero_scale_yml = config["aero_scale"].as<std::vector<double>>();
            m_aero_scale = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(aero_scale_yml.data(), 6);


            std::vector<double> x0_yml = config["ground_mdp_x0"].as<std::vector<double>>();
            std::vector<double> xd_yml = config["ground_mdp_xd"].as<std::vector<double>>();
            std::vector<double> X_yml = config["ground_mdp_X"].as<std::vector<double>>();
            std::vector<double> U_yml = config["ground_mdp_U"].as<std::vector<double>>();
            std::vector<double> Qx_yml = config["ground_mdp_Qx"].as<std::vector<double>>();
            std::vector<double> Qx_equ_yml = config["ground_mdp_Qx_equ"].as<std::vector<double>>();
            std::vector<double> Qu_yml = config["ground_mdp_Qu"].as<std::vector<double>>();
            std::vector<double> Qf_yml = config["ground_mdp_Qf"].as<std::vector<double>>();

            if (m_verbose) { std::cout << "setting lims ok" <<  std::endl;}

            assert(x0_yml.size() == m_state_dim);
            assert(xd_yml.size() == m_state_dim);
            assert(X_yml.size() == m_state_dim * 2);
            assert(U_yml.size() == m_action_dim * 2);
            assert(Qx_yml.size() == m_state_dim);
            assert(Qx_equ_yml.size() == m_state_dim);
            assert(Qu_yml.size() == m_action_dim);
            assert(Qf_yml.size() == m_state_dim);
            
            m_x0 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(x0_yml.data(), x0_yml.size());
            m_xd = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(xd_yml.data(), xd_yml.size());
            m_Qx = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(Qx_yml.data(), Qx_yml.size()).asDiagonal();
            m_Qx_equ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(Qx_equ_yml.data(), Qx_equ_yml.size()).asDiagonal();
            Eigen::MatrixXd Qu = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(Qu_yml.data(), Qu_yml.size()).asDiagonal();
            m_Qf = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(Qf_yml.data(), Qf_yml.size()).asDiagonal();
            m_X  = Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(X_yml.data(), m_state_dim, 2);
            Eigen::MatrixXd U = Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(U_yml.data(), m_action_dim, 2);
            

            m_Qx_inv = Eigen::MatrixXd::Zero(m_state_dim, m_state_dim);
            for (int ii=0; ii<m_state_dim; ii++) {
                if (m_Qx(ii,ii) > 1e-6) {
                    m_Qx_inv(ii,ii) = 1.0 / m_Qx(ii,ii);
                }
            }

            if (m_verbose) { std::cout << "setting weights ok" <<  std::endl;}

            // adjust action dimension based on flight mode
            if (m_flight_mode == 0) {
                // u = [delta_e, delta_r, delta_a]
                m_action_dim = 3; 
                m_Qu = Qu.block(0,0,3,3).eval();
                m_U = U.block(0,0,3,2).eval(); 
            } else if (m_flight_mode == 1) {
                // u = [thrust_z, tau_x, tau_y, tau_z]
                m_action_dim = 4; 
                m_Qu = Qu.block(3,3,4,4).eval();
                m_U = U.block(3,0,4,2).eval();
            } else if (m_flight_mode == 2) {
                // u = [delta_e, delta_r, delta_a, thrust_z, tau_x, tau_y, tau_z, thrust_x]
                m_action_dim = 8;
                m_Qu = Qu;
                m_U = U; 
            }

            // std::cout << "m_U" << std::endl; 
            // print_m(m_U);
            
            if (m_verbose) { std::cout << "setting flight mode ok" <<  std::endl;}

            // obstacle treatment is we add them dynamically through bindings
            m_special_obstacle_idxs = config["ground_mdp_special_obstacle_idxs"].as<std::vector<int>>();
            m_special_obstacle_radius = config["ground_mdp_special_obstacle_radius"].as<double>();
            m_obstacles.clear();

            if (m_verbose) { std::cout << "obstacle init ok" <<  std::endl;}
        }

        // standard mdp ...
        int state_dim() override { return m_state_dim; }

        int action_dim() override { return m_action_dim; }

        double dt() override { return m_dt; }
        
        int timestep_idx() override { return state_dim()-1; }

        Eigen::VectorXd initial_state() override { return m_x0; }

        Eigen::MatrixXd compute_dynamic_obstacle(const Eigen::MatrixXd & obstacle, int timestep) {
            Eigen::MatrixXd dynamic_obstacle = obstacle; // copy
            double duration = (timestep - obstacle(timestep_idx(),0)) * dt();
            Eigen::Vector3d dynamic_obstacle_center = {
                (obstacle(0,0) + obstacle(0,1)) / 2 + obstacle(3,0) * duration,
                (obstacle(1,0) + obstacle(1,1)) / 2 + obstacle(4,0) * duration,
                (obstacle(2,0) + obstacle(2,1)) / 2 + obstacle(5,0) * duration
            };
            dynamic_obstacle(0,0) = dynamic_obstacle_center(0) - (obstacle(0,1) - obstacle(0,0))/2; 
            dynamic_obstacle(0,1) = dynamic_obstacle_center(0) + (obstacle(0,1) - obstacle(0,0))/2;
            dynamic_obstacle(1,0) = dynamic_obstacle_center(1) - (obstacle(1,1) - obstacle(1,0))/2; 
            dynamic_obstacle(1,1) = dynamic_obstacle_center(1) + (obstacle(1,1) - obstacle(1,0))/2;
            dynamic_obstacle(2,0) = dynamic_obstacle_center(2) - (obstacle(2,1) - obstacle(2,0))/2;
            dynamic_obstacle(2,1) = dynamic_obstacle_center(2) + (obstacle(2,1) - obstacle(2,0))/2;
            return dynamic_obstacle;
        }

        bool is_state_valid(const Eigen::VectorXd & state) override { 

            // obstacles and state lims 
            for (int ii=0; ii<m_obstacles.size(); ii++) {
                if (std::find(m_special_obstacle_idxs.begin(), m_special_obstacle_idxs.end(), ii) != m_special_obstacle_idxs.end()) {
                    if (is_vec_in_cube(state.head(3), compute_dynamic_obstacle(m_obstacles[ii], state(timestep_idx())).block(0, 0, 3, 2))) { 
                        return false; 
                    }
                } else {
                    if (is_vec_in_cube(state.head(3), m_obstacles[ii].block(0, 0, 3, 2))) { 
                        return false; 
                    }
                }
            }

            if (!is_vec_in_cube(state, m_X)) { 
                if (m_verbose) { std::cout << "not valid because out of limits" << std::endl; }
                return false; 
            }

            // // wind angles 
            // double alpha = calc_alpha(state);
            // double beta = calc_beta(state);
            // if (std::abs(alpha) > m_max_alpha or std::abs(beta) > m_max_beta) {
            //     return false;
            // }

            return true; 
        }
        
        void set_xd(Eigen::VectorXd _xd) override {
            assert(_xd.size() == m_state_dim);
            m_xd = _xd;
        }
        void set_x0(Eigen::VectorXd _x0) override {
            assert(_x0.size() == m_state_dim);
            m_x0 = _x0;
        }

        void set_dt(double dt) override {
            m_dt = dt;
        }

        Eigen::Matrix<double,-1,2> X() override { return m_X; }

        Eigen::Matrix<double,-1,2> U() override { return m_U; }

        Eigen::VectorXd empty_control() override { 
            // return Eigen::VectorXd::Zero(action_dim()); 
            Eigen::VectorXd empty_u = (m_U.col(0) + m_U.col(1))/2.0;
            if (m_flight_mode == 1){
                double gravity_compensation = m_gravity * m_mass;
                if (gravity_compensation >= m_U(0,0) && gravity_compensation <= m_U(0,1)) {
                    empty_u(0,0) = gravity_compensation;
                }
            }
            return empty_u; 
        }

        // unique to SixDOFAircraft
        Eigen::VectorXd F(const Eigen::VectorXd & state, 
                          const Eigen::VectorXd & action) override {
            Eigen::VectorXd curr_state = state; 
            if (state.rows() == timestep_idx()+1){
                for (int ii=0; ii<m_control_hold; ii++) {
                    Eigen::VectorXd _dxdt = dxdt(curr_state, action);
                    curr_state = curr_state + _dxdt * m_dt;
                }
                curr_state(timestep_idx()) = state(timestep_idx()) + 1;
            } else if (state.rows() == timestep_idx()){
                for (int ii=0; ii<m_control_hold; ii++) {
                    Eigen::VectorXd _dxdt = dxdt(curr_state, action);
                    curr_state = curr_state + _dxdt.head(12) * m_dt;
                }
            } else {
                throw std::logic_error("state dim not expected ");
            }
            return curr_state;
        } 

            
        Eigen::VectorXd dxdt(const Eigen::VectorXd & state, 
                             const Eigen::VectorXd & action) {
            // x = [rx, ry, rz, u, v, w, phi, theta, psi, p, q, r, t] 
            // u = [delta_e, delta_r, delta_a, thrust_z, tau_x, tau_y, tau_z, thrust_x]
            double delta_e = 0.0;
            double delta_r = 0.0;
            double delta_a = 0.0;
            double thrust_z = 0.0;
            double tau_x = 0.0;
            double tau_y = 0.0;
            double tau_z = 0.0;
            double thrust_x = 0.0;
            if (m_flight_mode == 0) {
                delta_e = action(0,0);
                delta_r = action(1,0);
                delta_a = action(2,0);
            } else if (m_flight_mode == 1) {
                thrust_z = action(0,0);
                tau_x = action(1,0);
                tau_y = action(2,0);
                tau_z = action(3,0);
            } else if (m_flight_mode == 2) {
                delta_e = action(0,0);
                delta_r = action(1,0);
                delta_a = action(2,0);
                thrust_z = action(3,0);
                tau_x = action(4,0);
                tau_y = action(5,0);
                tau_z = action(6,0);
                thrust_x = action(7,0);
            } 

            // get rotation matrices 
            Eigen::MatrixXd R_ub = rot_mat_body_to_inertial(state(6,0) , state(7,0) , state(8,0) );

            // compute forces (in body frame) and moments about body frame axis
            // - gravity
            Eigen::Matrix<double,3,1> f_g; 
            f_g << 0.0, 0.0, m_mass * m_gravity; 
            f_g = R_ub.transpose() * f_g;

            // - thrust
            Eigen::Matrix<double,3,1> f_th; 
            f_th << thrust_x, 0.0, -thrust_z;

            Eigen::Matrix<double,3,1> moment_th;
            moment_th << tau_x, tau_y, tau_z;

            // - aero
            Eigen::Matrix<double,6,1> aero = aero_model(state, action);
            
            // wind hack - thermal enters as forces (and moments)
            if (m_wind_mode == 1) {
                Eigen::Matrix<double,3,1> f_thermal;
                Eigen::Matrix<double,3,1> tau_thermal;
                for (int ii=0; ii<m_Xs_thermal.size(); ii++) {
                    if (is_vec_in_cube(state.head(3), m_Xs_thermal[ii].block(0,0,3,2))) {
                        if (m_Vs_thermal[ii].rows() == 3) { // thermal just has forces
                            throw std::logic_error("Thermal force vector is three - old config!");
                        } else if (m_Vs_thermal[ii].rows() == 6) { // thermal has forces and moments
                            f_thermal = R_ub.transpose() * m_Vs_thermal[ii].block(0,0,3,1);
                            aero.block(0,0,3,1) += f_thermal;
                            tau_thermal = m_Vs_thermal[ii].block(3,0,3,1);
                            aero.block(3,0,3,1) += tau_thermal;
                        } else {
                            throw std::logic_error("thermal vector size not recognized");
                        }                   
                    } 
                }
            } else if (m_wind_mode == 2) {
                Eigen::Matrix<double,3,1> f_thermal;
                for (int ii=0; ii<m_Xs_thermal.size(); ii++) {
                    // if (is_vec_in_cube(state.head(3), m_Xs_thermal[ii].block(0,0,3,2)) 
                    //         && int(state(timestep_idx())) % m_wind_period < m_wind_duty_cycle * m_wind_period) {
                    if (is_vec_in_cube(state.head(3), m_Xs_thermal[ii].block(0,0,3,2))){
                        if (m_Vs_thermal[ii].rows() == 3) { // thermal just has forces
                            throw std::logic_error("Thermal force vector is three - old config!");
                        } else if (m_Vs_thermal[ii].rows() == 6) { // thermal has forces and moments
                            f_thermal = R_ub.transpose() * m_Vs_thermal[ii].block(0,0,3,1);
                            aero.block(0,0,3,1) += f_thermal;
                        } else {
                            throw std::logic_error("thermal vector size not recognized");
                        }
                    } 
                }
            }
            // std::cout << "aero: "; print_m(aero);

            // - total forces and moments 
            Eigen::Matrix<double,3,1> forces = f_g + f_th + aero.block(0,0,3,1);
            Eigen::Matrix<double,3,1> moments = moment_th + aero.block(3,0,3,1);

            // attitude derivative matrix 
            Eigen::Matrix<double,3,3> Y;
            Y << 1, sin(state(6,0) ) * tan(state(7,0) ), cos(state(6,0) ) * tan(state(7,0) ),
                 0, cos(state(6,0) ), - sin(state(6,0) ), 
                 0, sin(state(6,0) ) / cos(state(7,0) ), cos(state(6,0) ) / cos(state(7,0) );

            // state derivative
            Eigen::Matrix<double,13,1> _dxdt(state_dim());
            _dxdt.block(0,0,3,1) = R_ub * state.block(3,0,3,1);
            _dxdt(3) = forces(0) / m_mass - state(10,0)  * state(5,0)  + state(11,0)  * state(4,0) ; 
            _dxdt(4) = forces(1) / m_mass - state(11,0)  * state(3,0)  + state(9,0)  * state(5,0) ; 
            _dxdt(5) = forces(2) / m_mass - state(9,0)  * state(4,0)  + state(10,0)  * state(3,0) ; 

            _dxdt.block(6,0,3,1) = Y * state.block(9,0,3,1);

            double gamma = m_Ixx * m_Izz - m_Ixz * m_Ixz; 
            double gamma1 = (m_Ixz * (m_Ixx - m_Iyy + m_Izz)) / gamma; 
            double gamma2 = (m_Izz * (m_Izz - m_Iyy) + m_Ixz * m_Ixz) / gamma; 
            double gamma3 = m_Izz / gamma; 
            double gamma4 = m_Ixz / gamma; 
            double gamma5 = (m_Izz - m_Ixx) / m_Iyy; 
            double gamma6 = m_Ixz / m_Iyy;
            double gamma7 = (m_Ixx * (m_Ixx - m_Iyy) + m_Ixz * m_Ixz) / gamma;
            double gamma8 = m_Ixx / gamma;

            _dxdt(9)  = gamma1 * state(9,0)  * state(10,0)  - gamma2 * state(10,0)  * state(11,0)  + gamma3 * moments(0) + gamma4 * moments(2);
            _dxdt(10) = gamma5 * state(9,0)  * state(11,0)  - gamma6 * (state(9,0)  * state(9,0)  - state(11,0)  * state(11,0) ) + 1.0 / m_Iyy * moments(1);
            _dxdt(11) = gamma7 * state(9,0)  * state(10,0)  - gamma1 * state(10,0)  * state(11,0)  + gamma4 * moments(0) + gamma8 * moments(2);
            _dxdt(12) = 0.0;

            if (m_verbose) {
                std::cout << "F(state, action)" << std::endl;
                std::cout << "flight_mode: " << m_flight_mode << std::endl;
                std::cout << "state: state(9,0) x: " << state(0,0) << std::endl;
                std::cout << "state: state(9,0) y: " << state(1,0) << std::endl;
                std::cout << "state: state(9,0) z: " << state(2,0) << std::endl;
                std::cout << "state: state(3,0) : " << state(3,0)  << std::endl;
                std::cout << "state: state(4,0) : " << state(4,0)  << std::endl;
                std::cout << "state: state(5,0) : " << state(5,0)  << std::endl;
                std::cout << "state: state(6,0) : " << state(6,0)  << std::endl;
                std::cout << "state: state(7,0) : " << state(7,0)  << std::endl;
                std::cout << "state: state(8,0) : " << state(8,0)  << std::endl;
                std::cout << "state: state(9,0) : " << state(9,0)  << std::endl;
                std::cout << "state: state(10,0) : " << state(10,0)  << std::endl;
                std::cout << "state: state(11,0) : " << state(11,0)  << std::endl;
                std::cout << "action: delta_r: " << delta_r << std::endl;
                std::cout << "action: delta_a: " << delta_a << std::endl;
                std::cout << "action: thrust_z: " << thrust_z << std::endl;
                std::cout << "action: tau_x: " << tau_x << std::endl;
                std::cout << "action: tau_y: " << tau_y << std::endl;
                std::cout << "action: tau_z: " << tau_z << std::endl;
                std::cout << "action: thrust_x: " << thrust_x << std::endl;
                std::cout << "aero: "; print_v(aero);
                std::cout << "forces: "; print_v(forces);
                std::cout << "moments: "; print_v(moments);
                std::cout << "dxdt: "; print_v(_dxdt); 
            }

            return _dxdt;
        }

        Eigen::MatrixXd B(const Eigen::VectorXd & state) override { 
            double gamma = m_Ixx * m_Izz - m_Ixz * m_Ixz; 
            double gamma1 = (m_Ixz * (m_Ixx - m_Iyy + m_Izz)) / gamma; 
            double gamma2 = (m_Izz * (m_Izz - m_Iyy) + m_Ixz * m_Ixz) / gamma; 
            double gamma3 = m_Izz / gamma; 
            double gamma4 = m_Ixz / gamma; 
            double gamma5 = (m_Izz - m_Ixx) / m_Iyy; 
            double gamma6 = m_Ixz / m_Iyy;
            double gamma7 = (m_Ixx * (m_Ixx - m_Iyy) + m_Ixz * m_Ixz) / gamma;
            double gamma8 = m_Ixx / gamma;

            Eigen::MatrixXd _B = Eigen::MatrixXd::Zero(12,4); 
            _B(5,0) = 1 / m_mass;
            _B(9,1) = gamma3;
            _B(9,3) = gamma4;
            _B(10,2) = 1 / m_Iyy;
            _B(11,1) = gamma4;
            _B(11,3) = gamma8;
            return _B;
        }


        // aero stuff 
        Eigen::VectorXd wind_model(const Eigen::VectorXd & state) {
            if (m_wind_mode == 0) {
                return Eigen::VectorXd::Zero(3);
            } else if (m_wind_mode == 1) {
                return Eigen::VectorXd::Zero(3);
            } else {
                throw std::logic_error("wind model not implemented");
            } }


        double sign(double x){
            if (x > 0) { return 1.0; } 
            else { return -1.0; } }


        double sigmoid(double alpha) {
            return (1.0 + exp(-m_M * (alpha - m_alpha_0)) + exp(m_M * (alpha + m_alpha_0))) / (
                (1.0 + exp(-m_M * (alpha - m_alpha_0))) * (1.0 + exp(m_M * (alpha + m_alpha_0)))); }

        
        Eigen::Matrix<double,6,1> compute_aero_forces_and_moments_from_state_diff(
                Eigen::Matrix<double,13,1> actual_state, Eigen::Matrix<double,13,1> predicted_state, double dt) {
            Eigen::Matrix<double,6,1> aero_forces_and_moments;
            Eigen::Matrix<double,13,1> dxdotdt = (actual_state - predicted_state) / dt;
            double gamma = m_Ixx * m_Izz - m_Ixz * m_Ixz; 
            double gamma3 = m_Izz / gamma; 
            double gamma4 = m_Ixz / gamma; 
            double gamma8 = m_Ixx / gamma;
            aero_forces_and_moments(0,0) = m_mass * dxdotdt(3,0);
            aero_forces_and_moments(1,0) = m_mass * dxdotdt(4,0);
            aero_forces_and_moments(2,0) = m_mass * dxdotdt(5,0);
            aero_forces_and_moments(4,0) = m_Iyy * dxdotdt(10,0);
            aero_forces_and_moments(3,0) = (gamma8 * dxdotdt(9,0) - gamma4 * dxdotdt(11,0)) / (gamma3*gamma8 - gamma4*gamma4);
            aero_forces_and_moments(5,0) = (-gamma4 * dxdotdt(9,0) + gamma3 * dxdotdt(11,0)) / (gamma3*gamma8 - gamma4*gamma4);
            return aero_forces_and_moments;
        }


        double calc_alpha(const Eigen::VectorXd & state) {
            Eigen::VectorXd V_wu = wind_model(state); // absolute
            Eigen::VectorXd V_au = state.block(3,0,3,1) - V_wu; // relative
            double rel_wind_speed = V_au.norm();
            double alpha = atan2(V_au(2), V_au(0)) + 3.14 / 180 * 15.0; // add wing angle 
            alpha = std::max(std::min(alpha, m_max_alpha), -m_max_alpha);
            return alpha; 
        }


        double calc_beta(const Eigen::VectorXd & state) {
            Eigen::VectorXd V_wu = wind_model(state); // absolute
            Eigen::VectorXd V_au = state.block(3,0,3,1) - V_wu; // relative
            double rel_wind_speed = V_au.norm();
            double beta = atan2(V_au(1), sqrt(square(V_au(0)) + square(V_au(2))));
            beta = std::max(std::min(beta, m_max_beta), -m_max_beta);
            return beta; 
        }


        std::array<double,6> compute_aero_coeffs(const Eigen::VectorXd & state, const Eigen::VectorXd & action) {
            // x = [rx, ry, rz, u, v, w, phi, theta, psi, p, q, r, t] 
            // u = [delta_e, delta_r, delta_a, thrust_z, tau_x, tau_y, tau_z, thrust_x]
            double delta_e = 0.0;
            double delta_r = 0.0;
            double delta_a = 0.0;
            double thrust_z = 0.0;
            double tau_x = 0.0;
            double tau_y = 0.0;
            double tau_z = 0.0;
            double thrust_x = 0.0;
            if (m_flight_mode == 0) {
                delta_e = action(0,0);
                delta_r = action(1,0);
                delta_a = action(2,0);
            } else if (m_flight_mode == 1) {
                thrust_z = action(0,0);
                tau_x = action(1,0);
                tau_y = action(2,0);
                tau_z = action(3,0);
            } else if (m_flight_mode == 2) {
                delta_e = action(0,0);
                delta_r = action(1,0);
                delta_a = action(2,0);
                thrust_z = action(3,0);
                tau_x = action(4,0);
                tau_y = action(5,0);
                tau_z = action(6,0);
                thrust_x = action(7,0);
            } 

            // get wind speed and angles 
            Eigen::VectorXd V_wu = wind_model(state); // absolute
            Eigen::VectorXd V_au = state.block(3,0,3,1) - V_wu; // relative
            double rel_wind_speed = V_au.norm();
            // double alpha = atan2(V_au(2), V_au(0)) + 3.14 / 180 * 15.0; // add wing angle 
            // // double alpha = atan2(V_au(2), V_au(0)) + 3.14 / 180 * 5.0; // add wing angle 
            // double beta = atan2(V_au(1), sqrt(square(V_au(0)) + square(V_au(2))));

            // alpha = std::max(std::min(alpha, m_max_alpha), -m_max_alpha);
            // beta = std::max(std::min(beta, m_max_beta), -m_max_beta);

            double alpha = calc_alpha(state);
            double beta = calc_beta(state);

            // when starting from zero velocity (i.e. takeoff), we get nans because alpha and beta have zero denominators. 
            // I think anything that avoids nans coming from this is fine from a physics perspective, because aero forces get multiplied by velocity later 
            if (isnan(alpha)) { alpha = 0.0; }
            if (isnan(beta)) { beta = 0.0; }

            // AeroCoeffs aero_coeffs;
            std::array<double,6> aero_coeffs;
            if (rel_wind_speed == 0.0) { 
                ; 
            } else if (m_aero_mode == 0) {
                aero_coeffs = {0,0,0,0,0,0};
            } else if (m_aero_mode == 1 || m_aero_mode == 6) {
                aero_coeffs[0] = m_C_D_0 + m_C_D_alpha * alpha + m_C_D_q * m_c / (2 * rel_wind_speed) * state(10,0)  + m_C_D_delta_e * delta_e;
                aero_coeffs[1] = m_C_L_0 + m_C_L_alpha * alpha + m_C_L_q * m_c / (2 * rel_wind_speed) * state(10,0)  + m_C_L_delta_e * delta_e;
                aero_coeffs[2] = m_C_M_0 + m_C_M_alpha * alpha + m_C_M_q * m_c / (2 * rel_wind_speed) * state(10,0)  + m_C_M_delta_e * delta_e;
                aero_coeffs[3] = m_C_Y_0 + m_C_Y_beta * beta + m_C_Y_p * m_b / (2 * rel_wind_speed) * state(9,0)  + m_C_Y_r * m_b / (2 * rel_wind_speed) * state(11,0)  + m_C_Y_delta_a * delta_a + m_C_Y_delta_r * delta_r;
                aero_coeffs[4] = m_C_l_0 + m_C_l_beta * beta + m_C_l_p * m_b / (2 * rel_wind_speed) * state(9,0)  + m_C_l_r * m_b / (2 * rel_wind_speed) * state(11,0)  + m_C_l_delta_a * delta_a + m_C_l_delta_r * delta_r; 
                aero_coeffs[5] = m_C_n_0 + m_C_n_beta * beta + m_C_n_p * m_b / (2 * rel_wind_speed) * state(9,0)  + m_C_n_r * m_b / (2 * rel_wind_speed) * state(11,0)  + m_C_n_delta_a * delta_a + m_C_n_delta_r * delta_r;
            } else if (m_aero_mode == 2) {
                // clip alpha, then linear regime 
                alpha = std::min(std::max(alpha, -m_alpha_0), m_alpha_0);
                aero_coeffs[0] = m_C_D_p + m_C_D_0 + m_C_D_alpha * alpha + m_C_D_q * m_c / (2 * rel_wind_speed) * state(10,0)  + m_C_D_delta_e * delta_e;
                aero_coeffs[1] = m_C_L_0 + m_C_L_alpha * alpha + m_C_L_q * m_c / (2 * rel_wind_speed) * state(10,0)  + m_C_L_delta_e * delta_e;
                aero_coeffs[2] = m_C_M_0 + m_C_M_alpha * alpha + m_C_M_q * m_c / (2 * rel_wind_speed) * state(10,0)  + m_C_M_delta_e * delta_e;
                aero_coeffs[3] = m_C_Y_0 + m_C_Y_beta * beta + m_C_Y_p * m_b / (2 * rel_wind_speed) * state(9,0)  + m_C_Y_r * m_b / (2 * rel_wind_speed) * state(11,0)  + m_C_Y_delta_a * delta_a + m_C_Y_delta_r * delta_r;
                aero_coeffs[4] = m_C_l_0 + m_C_l_beta * beta + m_C_l_p * m_b / (2 * rel_wind_speed) * state(9,0)  + m_C_l_r * m_b / (2 * rel_wind_speed) * state(11,0)  + m_C_l_delta_a * delta_a + m_C_l_delta_r * delta_r; 
                aero_coeffs[5] = m_C_n_0 + m_C_n_beta * beta + m_C_n_p * m_b / (2 * rel_wind_speed) * state(9,0)  + m_C_n_r * m_b / (2 * rel_wind_speed) * state(11,0)  + m_C_n_delta_a * delta_a + m_C_n_delta_r * delta_r;
            } else {
                throw std::logic_error("aero coeffs not implemented");
            } 
            return aero_coeffs; 
        }


        // Eigen::VectorXd aero_model(
        Eigen::Matrix<double,6,1> aero_model(const Eigen::VectorXd & state, const Eigen::VectorXd & action) {
            
            Eigen::Matrix<double,6,1> aero;

            if (m_aero_mode == 3) {
                Eigen::VectorXd input(state.size() - 1 + action.size());
                input << state.head(m_state_dim-1), action;
                // aero = m_ff.eval(input);
            } else if (m_aero_mode == 4) {
                aero << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                // check if in thermal, if so add thermal forces and moments with (x,y) offset by center distance
                for (int ii=0; ii<m_Xs_thermal.size(); ii++) {
                    if (is_vec_in_cube(state.head(3), m_Xs_thermal[ii].block(0,0,3,2))) {
                        Eigen::VectorXd offset_state = state;
                        offset_state(0,0) -= (m_Xs_thermal[ii](0,0) + m_Xs_thermal[ii](0,1)) / 2.0;
                        offset_state(1,0) -= (m_Xs_thermal[ii](1,0) + m_Xs_thermal[ii](1,1)) / 2.0;
                        Eigen::VectorXd input(offset_state.size() - 1 + 1 + action.size());
                        input << offset_state.head(m_state_dim-1), 20.0, action;
                        // aero += m_ff.eval(input);
                    }
                }
            } else if (m_aero_mode == 5) {
                aero << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                // check if in thermal, if so add thermal forces and moments with (x,y) offset by center distance
                for (int ii=0; ii<m_Xs_thermal.size(); ii++) {
                    if (is_vec_in_cube(state.head(3), m_Xs_thermal[ii].block(0,0,3,2))) {
                        Eigen::VectorXd offset_state = state;
                        offset_state(0,0) -= (m_Xs_thermal[ii](0,0) + m_Xs_thermal[ii](0,1)) / 2.0;
                        offset_state(1,0) -= (m_Xs_thermal[ii](1,0) + m_Xs_thermal[ii](1,1)) / 2.0;
                        Eigen::VectorXd input(offset_state.size() - 1 + action.size());
                        // input << offset_state.head(m_state_dim-1), action;
                        input << offset_state.head(3), Eigen::VectorXd::Zero(9), 0.294, 0.0, 0.0, 0.0;
                        // aero.tail(3) += m_neural_thermal_scale * m_ff.eval(input).tail(3);
                        // std::cout << "input: " << input.transpose() << std::endl;
                        // std::cout << "aero: " << aero.transpose() << std::endl;
                        // Eigen::VectorXd check_input(16);
                        // check_input << 0.0, 0.0, -2.0, Eigen::VectorXd::Zero(9), 0.294, 0.0, 0.0, 0.0;
                        // std::cout << "check_input: " << check_input.transpose() << std::endl;
                        // std::cout << "check aero: " << m_ff.eval(check_input).transpose() << std::endl;

                    }
                }
            } else if (m_aero_mode == 0) {
                aero << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            } else {
                std::array<double,6> aero_coeffs = compute_aero_coeffs(state, action);

                // // get wind speed and angles 
                Eigen::VectorXd V_wu = wind_model(state); // absolute
                Eigen::VectorXd V_au = state.block(3,0,3,1) - V_wu; // relative
                double rel_wind_speed = V_au.norm();
                // // double alpha = atan2(V_au(2), V_au(0)) + 3.14 / 180 * 5.0; // add wing angle 
                // double alpha = atan2(V_au(2), V_au(0)) + 3.14 / 180 * 15.0; // add wing angle 
                // double beta = atan2(V_au(1), sqrt(square(V_au(0)) + square(V_au(2))));
                // alpha = std::max(std::min(alpha, m_max_alpha), -m_max_alpha);
                // beta = std::max(std::min(beta, m_max_beta), -m_max_beta);

                double alpha = calc_alpha(state);
                double beta = calc_beta(state);

                // when starting from zero velocity (i.e. takeoff), we get nans because alpha and beta have zero denominators. 
                // I think anything that avoids nans coming from this is fine from a physics perspective, because aero forces get multiplied by velocity later 
                if (isnan(alpha)) { alpha = 0.0; }
                if (isnan(beta)) { beta = 0.0; }

                double f_lift = 0.5 * m_rho * square(rel_wind_speed) * m_S * aero_coeffs[1];
                double f_drag = 0.5 * m_rho * square(rel_wind_speed) * m_S * aero_coeffs[0];

                f_drag *= m_obs_min_speed;

                if (m_aero_mode == 6 && is_vec_in_cube(state.head(3), m_Xs_thermal[0].block(0,0,3,2))) {
                    f_lift *= m_oswald_eff; 
                }

                double f_xa = - cos(alpha) * f_drag + sin(alpha) * f_lift;
                double f_ya = 0.5 * m_rho * square(rel_wind_speed) * m_S * aero_coeffs[3];
                double f_za = - sin(alpha) * f_drag - cos(alpha) * f_lift;
                double l_a = 0.5 * m_rho * square(rel_wind_speed) * m_S * m_b * aero_coeffs[4];
                double m_a = 0.5 * m_rho * square(rel_wind_speed) * m_S * m_c * aero_coeffs[2];
                double n_a = 0.5 * m_rho * square(rel_wind_speed) * m_S * m_b * aero_coeffs[5];
                
                aero << f_xa, f_ya, f_za, l_a, m_a, n_a;
            }

            // // clip aero forces 
            aero(0,0) = std::max(std::min(aero(0,0), m_aero_clip(0,1)), m_aero_clip(0,0));
            aero(1,0) = std::max(std::min(aero(1,0), m_aero_clip(1,1)), m_aero_clip(1,0));
            aero(2,0) = std::max(std::min(aero(2,0), m_aero_clip(2,1)), m_aero_clip(2,0));
            aero(3,0) = std::max(std::min(aero(3,0), m_aero_clip(3,1)), m_aero_clip(3,0));
            aero(4,0) = std::max(std::min(aero(4,0), m_aero_clip(4,1)), m_aero_clip(4,0));
            aero(5,0) = std::max(std::min(aero(5,0), m_aero_clip(5,1)), m_aero_clip(5,0));

            // scale aero forces
            aero(0,0) = m_aero_scale(0) * aero(0,0);
            aero(1,0) = m_aero_scale(1) * aero(1,0);
            aero(2,0) = m_aero_scale(2) * aero(2,0);
            aero(3,0) = m_aero_scale(3) * aero(3,0);
            aero(4,0) = m_aero_scale(4) * aero(4,0);
            aero(5,0) = m_aero_scale(5) * aero(5,0);

            // std::cout << "aero: " << std::endl;
            // print_v(aero);

            return aero; 
        }

        // void set_weights(std::vector<Eigen::MatrixXd> weightss, std::vector<Eigen::MatrixXd> biass) override { 
        //     m_ff.set_weights(weightss, biass); 
        // }

        Eigen::VectorXd eval_ff(const Eigen::VectorXd & state, const Eigen::VectorXd & action) override {
            Eigen::VectorXd input(m_state_dim - 1 + m_action_dim);
            input << state.head(m_state_dim-1), action;
            // return m_ff.eval(input);
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

        Eigen::MatrixXd rot_mat_body_to_wind(double alpha, double beta) {
            Eigen::MatrixXd R(3,3); 
            R << cos(alpha) * cos(beta), sin(beta), sin(alpha) * cos(beta),
                - cos(alpha) * sin(beta), cos(beta), - sin(alpha) * sin(beta),
                - sin(alpha), 0, cos(alpha);
            return R; }

        bool is_unsafe(const Eigen::VectorXd & state, const Eigen::MatrixXd & obstacle) {
            Eigen::Matrix<double, 2, 1> unsafe_center {
                (obstacle(0,0) + obstacle(0,1))/2.0, 
                (obstacle(1,0) + obstacle(1,1))/2.0 };
            if ((state.head(2) - unsafe_center).norm() < m_special_obstacle_radius) {
                return true;
            } else {
                return false;
            }
        }

        double R(const Eigen::VectorXd & state, 
                const Eigen::VectorXd & action) override {
            double r = 0.0;
            if (m_reward_mode == 0) {
                
                Eigen::VectorXd normalized_x(m_state_dim);
                Eigen::VectorXd normalized_xd(m_state_dim);
                for (int ii = 0; ii < m_state_dim; ii++){
                    normalized_x(ii) = (state(ii) - m_X(ii,0)) / (m_X(ii,1) - m_X(ii,0));
                    normalized_xd(ii) = (m_xd(ii) - m_X(ii,0)) / (m_X(ii,1) - m_X(ii,0)); }
                
                r = 1.0 - (2.0 / M_PI * atan(((normalized_x - normalized_xd).transpose() * m_Qx * (normalized_x - normalized_xd)).value()));
                // r = 1.0 - std::exp(-1 * ((normalized_x - normalized_xd).transpose() * m_Qx * (normalized_x - normalized_xd)).value());


            } else if (m_reward_mode == 2) {
                
                float angle = compute_visibility_angle(m_xd, state);
                float dist = (m_xd.block(0,0,3,1) - state.block(0,0,3,1)).norm();

                for (int special_obstacle_idx : m_special_obstacle_idxs) {
                    if (is_unsafe(state, compute_dynamic_obstacle(m_obstacles[special_obstacle_idx], state(timestep_idx())))) {
                        double obstacle_euclidean_distance = (state.head(2) - (compute_dynamic_obstacle(m_obstacles[special_obstacle_idx], state(timestep_idx())).block(0, 0, 2, 1) + 
                            compute_dynamic_obstacle(m_obstacles[special_obstacle_idx], state(timestep_idx())).block(0, 1, 2, 1))/2.0).norm();
                        return 0.1 - 0.1 * (1 - obstacle_euclidean_distance / (m_special_obstacle_radius - 0.2));
                        // return 0.0;
                    }
                }

                r = 0.1;
                if (in_obs_cone(angle, dist)) {
                    if (line_of_sight(state, m_xd)) {
                        r += 0.7;
                        // r += 0.3;
                        // r += 0.4 * std::exp(-1 * abs(angle));
                        // r += 0.4 * std::exp(-1 * (abs(angle) + abs(0.67 * dist - m_obs_cone_length)));
                        r += 0.2 * std::exp(-1 * state.block(3,0,3,1).norm());
                        r += 0.2 * std::exp(-1 * (abs(angle) + abs(0.67 * dist - m_obs_cone_length)));
                    }
                }

            } else if (m_reward_mode == 3) {
                if ((state(2) < m_xd(2)+0.5) && (state(2) > m_xd(2)-0.5) && 
                        (state.block(3,0,2,1).norm() > m_xd(3,0))) {
                    r = 1.0;
                } else {
                    r = 0.0;
                }
            } 

            return r; 
        }


        bool in_obs_cone(double angle, double dist) {
            return abs(angle) < m_obs_cone_angle && dist < m_obs_cone_length;
        }


        // bool line_of_sight(Eigen::VectorXd const & state) {

        //     for (auto idx : m_special_obstacle_idxs) {

        //         // make geometry 
        //         // double dx = std::min(
        //         //     (m_obstacles[2](0,1) - m_obstacles[2](0,0) - 1.0e-6) / 2.0,
        //         //     (m_obstacles[2](1,1) - m_obstacles[2](1,0) - 1.0e-6) / 2.0
        //         //     );
        //         // double dx = 0.05;
        //         double dx = 0.1;
        //         double distance = (state.head(3) - m_xd.head(3)).norm();
        //         Eigen::VectorXd slope = (m_xd.head(3) - state.head(3)) / distance;

        //         // step
        //         double curr_distance = dx;
        //         while (curr_distance < distance) {
        //             if (is_vec_in_cube(state.head(3) + slope * curr_distance, 
        //                     compute_dynamic_obstacle(m_obstacles[idx], state(timestep_idx())).block(0,0,3,2))) {
        //                 return false;
        //             }
        //             curr_distance += dx;
        //         }
        //     }
        //     return true;
        // }

        bool line_of_sight(Eigen::VectorXd const & state, Eigen::VectorXd const & target) {

            for (int idx=0; idx<m_obstacles.size(); idx++) {

                double dx = 0.25;
                double distance = (state.head(3) - target.head(3)).norm();
                Eigen::VectorXd slope = (target.head(3) - state.head(3)) / distance;

                Eigen::MatrixXd curr_obstacle = m_obstacles[idx];
                if (std::find(m_special_obstacle_idxs.begin(), m_special_obstacle_idxs.end(), idx) != m_special_obstacle_idxs.end()) {
                    curr_obstacle = compute_dynamic_obstacle(m_obstacles[idx], state(timestep_idx()));
                }

                // step
                double curr_distance = dx;
                while (curr_distance < distance) {
                    if (is_vec_in_cube(state.head(3) + slope * curr_distance, curr_obstacle.block(0,0,3,2))) {
                        return false;
                    }
                    curr_distance += dx;
                }
            }
            return true;
        }


        // observation helper 
        double compute_visibility_angle(const Eigen::VectorXd & state_d, const Eigen::VectorXd & state) {
            // pretend camera sticks out of nose of aircraft
            Eigen::MatrixXd R_ub = rot_mat_body_to_inertial(state(6,0), state(7,0), state(8,0));
            Eigen::Matrix<double,3,1> u_body; 
            u_body << 1.0, 0.0, 0.0;
            auto u_world = R_ub * u_body;
            auto uvr = (state_d.block(0,0,3,1) - state.block(0,0,3,1)).normalized();
            double angle = acos((uvr.transpose() * u_world).value());
            return angle;
        }

        int H() override { return m_H; }

        double gamma() override { return m_gamma; }

        void add_obstacle(Eigen::Matrix<double,-1,2> obstacle) override { 
            m_obstacles.push_back(obstacle); }

        void clear_obstacles() override {
            m_obstacles.clear(); }

        void add_thermal(Eigen::MatrixXd X_thermal, Eigen::VectorXd V_thermal) override { 
            m_Xs_thermal.push_back(X_thermal); 
            m_Vs_thermal.push_back(V_thermal); 
        }

        void clear_thermals() override {
            m_Xs_thermal.clear(); 
            m_Vs_thermal.clear(); 
        }

        // leaf oracle for UCT 
        double V(const Eigen::VectorXd state) override {
            return m_V_alive;
            // return 3.0;
            // return 10.0;
        }

        double V(const Eigen::VectorXd state, RNG& rng) override {
            return V(state);
        }

        // jacobian information for SCP
        Eigen::MatrixXd dFdx(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return F(x, action);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(state, f, fjac, accuracy, 1.0e-5);
            return fjac; }

        Eigen::MatrixXd dFdu(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> Eigen::VectorXd {
                return F(state, u);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(action, f, fjac, accuracy, 1.0e-5);
            return fjac; }

        Eigen::VectorXd dRdx(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> double {
                return R(x, action);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::VectorXd fgrad;
            fd::finite_gradient(state, f, fgrad, accuracy, 1.0e-5);
            return fgrad; }

        Eigen::MatrixXd d2Rdx2(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> double {
                return R(x, action);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fhess;
            fd::finite_hessian(state, f, fhess, accuracy, 1.0e-5);
            return fhess; }

        Eigen::MatrixXd d2Rdx2_inv(Eigen::VectorXd state, Eigen::VectorXd action) override {
            Eigen::MatrixXd d2Rdx2_ = d2Rdx2(state, action);
            Eigen::MatrixXd d2Rdx2_inv_ = d2Rdx2_.completeOrthogonalDecomposition().pseudoInverse();
            return d2Rdx2_inv_; }

        Eigen::VectorXd dRdu(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> double {
                return R(state, u);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::VectorXd fgrad;
            fd::finite_gradient(action, f, fgrad, accuracy, 1.0e-5);
            return fgrad; }

        Eigen::VectorXd dVdx(Eigen::VectorXd state) override {
            const auto f = [&](const Eigen::VectorXd& x) -> double {
                return V(x);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::VectorXd fgrad;
            fd::finite_gradient(state, f, fgrad, accuracy, 1.0e-5);
            return fgrad; }

        Eigen::MatrixXd d2Rdu2(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> double {
                return R(state, u);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fhess;
            fd::finite_hessian(action, f, fhess, accuracy, 1.0e-5);
            return fhess; }

        Eigen::MatrixXd d2Vdx2(Eigen::VectorXd state) override {
            const auto f = [&](const Eigen::VectorXd& x) -> double {
                return V(x);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fhess;
            fd::finite_hessian(state, f, fhess, accuracy, 1.0e-5);
            return fhess; }

        Eigen::VectorXd get_xd() override { return m_xd; }

        Eigen::MatrixXd sqrtQx() override {
            return m_Qx.cwiseSqrt();
        }

        Eigen::MatrixXd sqrtQx_equ() override {
            return m_Qx_equ.cwiseSqrt();
        }

        Eigen::MatrixXd sqrtQu() override {
            return m_Qu.cwiseSqrt();
        }

        Eigen::MatrixXd sqrtQf() override {
            return m_Qf.cwiseSqrt();
        }


    private: 
        Eigen::VectorXd m_x0;
        Eigen::VectorXd m_xd;
        Eigen::Matrix<double,-1,2> m_X;
        Eigen::Matrix<double,-1,2> m_U;
        Eigen::MatrixXd m_Qx;
        Eigen::MatrixXd m_Qx_equ;
        Eigen::MatrixXd m_Qx_inv;
        Eigen::MatrixXd m_Qu;
        Eigen::MatrixXd m_Qf;
        // modes 
        int m_flight_mode;
        int m_wind_mode;
        int m_aero_mode;
        int m_reward_mode;
        bool m_verbose;
        // standard param 
        double m_V_alive; 
        double m_dt; 
        double m_gamma;
        int m_H; 
        int m_state_dim;
        int m_action_dim;
        int m_control_hold;
        // physics param 
        double m_mass;
        double m_gravity;
        double m_rho;
        double m_S;
        double m_b;
        double m_c;
        double m_Izz;
        double m_Ixx;
        double m_Izx;
        double m_Ixz;
        double m_Iyy;
        // thermals 
        std::vector<Eigen::MatrixXd> m_Xs_thermal;
        std::vector<Eigen::VectorXd> m_Vs_thermal;
        int m_wind_period;
        double m_wind_duty_cycle;
        // obstacles 
        std::vector<int> m_special_obstacle_idxs;
        double m_special_obstacle_radius;
        // observation model 
        double m_obs_cone_angle;
        double m_obs_cone_length;
        double m_obs_min_speed;
        // takeoff conditions
        double m_alpha_takeoff;
        double m_beta_takeoff;
        double m_speed_takeoff;
        // aero model
        Eigen::MatrixXd m_aero_clip;
        Eigen::VectorXd m_aero_scale;
        // neural 
        // FeedForwardNetwork m_ff; 
        // nonlinear lift / drag
        double m_alpha_0;
        double m_M;
        double m_C_D_p;
        double m_oswald_eff;
        double m_max_alpha;
        double m_max_beta;
        // linear 
        double m_C_D_0;
        double m_C_D_alpha;
        double m_C_D_q;
        double m_C_D_delta_e;
        double m_C_L_0;
        double m_C_L_alpha;
        double m_C_L_q;
        double m_C_L_delta_e;
        double m_C_M_0;
        double m_C_M_alpha;
        double m_C_M_q;
        double m_C_M_delta_e;
        double m_C_Y_0;
        double m_C_Y_beta;
        double m_C_Y_p;
        double m_C_Y_r;
        double m_C_Y_delta_a;
        double m_C_Y_delta_r;
        double m_C_l_0;
        double m_C_l_beta;
        double m_C_l_p;
        double m_C_l_r;
        double m_C_l_delta_a;
        double m_C_l_delta_r;
        double m_C_n_0;
        double m_C_n_beta;
        double m_C_n_p;
        double m_C_n_r;
        double m_C_n_delta_a;
        double m_C_n_delta_r;

        double m_neural_thermal_scale;
        std::vector<int> m_my_idxs;
};

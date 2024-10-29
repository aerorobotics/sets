

#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <math.h>
#include <yaml-cpp/yaml.h>

#include "mdp.hpp"
#include "sixdofaircraft.hpp"
#include "../util/util.hpp"


class GameSixDOFAircraft : public MDP {

    public:

        GameSixDOFAircraft(std::string config_path) {
            YAML::Node config = YAML::LoadFile(config_path);

            m_sixdof = new SixDOFAircraft(config_path);

            std::vector<std::vector<double>> targets_yml = config["targets"].as<std::vector<std::vector<double>>>();
            m_targets.clear();
            for (int ii=0; ii<targets_yml.size(); ii++){
                m_targets.push_back(Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(targets_yml[ii].data(), not_augmented_state_dim()));
            }
            m_special_target_idxs = config["target_special_idxs"].as<std::vector<int>>();

            m_reward_parameters = config["reward_parameters"].as<std::vector<double>>();
        }

        std::string name() override {
            return "GameSixDOFAircraft";
        }

        std::vector<int> velocity_idxs() override { return m_sixdof->velocity_idxs(); }

        std::vector<int> position_idxs() override { return m_sixdof->position_idxs(); }

        std::vector<int> my_idxs() override { return m_sixdof->my_idxs(); }

        // same as ground mdp 
        int state_dim() override { return m_sixdof->state_dim() + m_targets.size(); }

        int not_augmented_state_dim() override { return m_sixdof->state_dim(); }

        int action_dim() override { return m_sixdof->action_dim(); }

        int timestep_idx() override { return m_sixdof->timestep_idx(); }

        double dt() override { return m_sixdof->dt(); }

        Eigen::VectorXd initial_state() override { 
            Eigen::VectorXd x0(state_dim());
            x0.head(not_augmented_state_dim()) = m_sixdof->initial_state();
            for (int ii=0; ii<m_targets.size(); ii++) {
                x0(ii+not_augmented_state_dim()) = -100.0;
            }
            return x0; 
        }

        Eigen::VectorXd empty_control() override { return m_sixdof->empty_control(); }

        // obstacles
        void clear_obstacles() override { return m_sixdof->clear_obstacles(); }

        void add_obstacle(Eigen::Matrix<double,-1,2> obstacle) override { return m_sixdof->add_obstacle(obstacle); }

        std::vector<Eigen::MatrixXd> obstacles() override { return m_sixdof->obstacles(); }

        // thermals
        void clear_thermals() override { return m_sixdof->clear_thermals(); }

        void add_thermal(Eigen::MatrixXd X_thermal, Eigen::VectorXd V_thermal) override { return m_sixdof->add_thermal(X_thermal, V_thermal); }

        // targets
        void clear_targets() override { return m_targets.clear(); }

        void add_target(Eigen::VectorXd target) override { return m_targets.push_back(target); }

        bool is_state_valid(const Eigen::VectorXd & state) override { 
            return m_sixdof->is_state_valid(state.head(not_augmented_state_dim())); }

        Eigen::Matrix<double,-1,2> X() override { return m_sixdof->X(); }

        Eigen::Matrix<double,-1,2> U() override { return m_sixdof->U(); }

        Eigen::VectorXd F(const Eigen::VectorXd & state, 
                          const Eigen::VectorXd & action) override {
            Eigen::VectorXd next_state = state;
            next_state.head(not_augmented_state_dim()) = m_sixdof->F(state.head(not_augmented_state_dim()), action);
            Eigen::VectorXd augmented_state = update_augmented_state_only(next_state, action);
            next_state.tail(m_targets.size()) = augmented_state;
            return next_state;
        }

        Eigen::MatrixXd B(const Eigen::VectorXd & state) override {
            return m_sixdof->B(state);
        }

        Eigen::VectorXd F_non_augmented(const Eigen::VectorXd & state, const Eigen::VectorXd & action) override {
            return m_sixdof->F(state, action);
        }

        Eigen::VectorXd update_augmented_state_only(const Eigen::VectorXd & state, const Eigen::VectorXd & action) override {
        
            // std::cout << "m_targets.size(): " << m_targets.size() << std::endl;

            Eigen::VectorXd augmented_state(m_targets.size());

            // update "time since target viewed"
            for (int ii=0; ii<m_targets.size(); ii++) {

                // std::cout << "m_targets[ii]: " << std::endl;
                // print_v(m_targets[ii]);
                
                Eigen::VectorXd target = compute_dynamic_target(m_targets[ii], state(timestep_idx()));

                // print_v(target);

                double angle = m_sixdof->compute_visibility_angle(target, state);
                double dist = (target.head(3) - state.head(3)).norm();

                // if (m_sixdof->in_obs_cone(angle, dist) && m_sixdof->line_of_sight(state, target)) {
                if (m_sixdof->in_obs_cone(angle, dist)) {
                    augmented_state(ii) = 0.0;
                } else if (state(ii+not_augmented_state_dim()) < -0.1) {
                    augmented_state(ii) = state(ii+not_augmented_state_dim());
                } else {
                    augmented_state(ii) = state(ii+not_augmented_state_dim()) + 1.0;
                }
            }

            return augmented_state;
        }

        void set_weights(std::vector<Eigen::MatrixXd> weightss, std::vector<Eigen::MatrixXd> biass) override { 
            m_sixdof->set_weights(weightss, biass); 
        }

        void set_dt(double dt) override { 
            m_sixdof->set_dt(dt); 
        }

        Eigen::MatrixXd sqrtQx() override {
            return m_sixdof->sqrtQx();
        }

        Eigen::MatrixXd sqrtQx_equ() override {
            return m_sixdof->sqrtQx_equ();
        }
        
        Eigen::MatrixXd sqrtQu() override {
            return m_sixdof->sqrtQu();
        }

        Eigen::MatrixXd sqrtQf() override {
            return m_sixdof->sqrtQf();
        }



        double R(const Eigen::VectorXd & state, 
                const Eigen::VectorXd & action) override { 
            return R_verbose(state, action, false);
        }


        double R_verbose(const Eigen::VectorXd & state, 
                const Eigen::VectorXd & action, bool verbose) override { 

            double weight_alive = m_reward_parameters[0];
            double weight_target = m_reward_parameters[1];
            double timescale_target = m_reward_parameters[2];
            double weight_energy = m_reward_parameters[3];
            double lengthscale_target = m_reward_parameters[4];
            double weight_angle = m_reward_parameters[5];
            double scale_angle = m_reward_parameters[6];
    
            // std::cout << "start R" << std::endl;

            double reward_alive = 1.0;

            double reward_target = 0.0;
            for (int ii=0; ii<m_targets.size(); ii++) {
                double time_since_seen_target_ii = state(ii+not_augmented_state_dim());
                // this if condition handles the initialization of time_since_seen_target_ii=-100
                // if (time_since_seen_target_ii > -0.1) {
                if (true) {
                    if (time_since_seen_target_ii > 0 && time_since_seen_target_ii*timescale_target < 1) {
                        reward_target += 1.0 / m_targets.size();
                    } else {
                        double dist_to_target = (state.segment(0,3) - m_targets[ii].segment(0,3)).norm();
                        reward_target += 0.5 / m_targets.size() * dist_to_reward(dist_to_target, lengthscale_target); 
                        // reward_target += 0.0; 
                    }
                }
            }

            double potential_energy = 11.0 * 9.8 * (-1 * state(2,0)); // m g h 
            double kinetic_energy = 0.5 * 11.0 * std::pow(state.segment(3,3).norm(),2); // 1/2 m v^2
            double total_energy = potential_energy + kinetic_energy;
            double max_potential_energy = 11.0 * 9.8 * (-1 * m_sixdof->X()(2,0));
            double max_kinetic_energy = 0.5 * 11.0 * 30.0 * 30.0;
            double max_total_energy = max_potential_energy + max_kinetic_energy;
            double reward_energy = std::min(total_energy / max_total_energy, 1.0);
    
            double dist_angle = state.segment(6,2).norm();
            double reward_angle = dist_to_reward(dist_angle, scale_angle);

            double reward = 
                weight_alive * reward_alive + 
                weight_target * reward_target + 
                weight_energy * reward_energy +
                weight_angle * reward_angle;

            if (verbose) {
                std::cout << "weight_alive * reward_alive: " << weight_alive * reward_alive << std::endl;
                std::cout << "weight_target * reward_target: " << weight_target * reward_target << std::endl;
                std::cout << "weight_energy * reward_energy: " << weight_energy * reward_energy << std::endl;
                std::cout << "weight_angle * reward_angle: " << weight_angle * reward_angle << std::endl;
                // std::cout << "potential_energy: " << potential_energy << std::endl;
                // std::cout << "kinetic_energy: " << kinetic_energy << std::endl;
                // std::cout << "total_energy: " << total_energy << std::endl;
                // std::cout << "max_total_energy: " << max_total_energy << std::endl;
                std::cout << "reward: " << reward << std::endl;
            }

            return reward; 
        }

        double dist_to_reward(double dist, double scale) {
            return 1.0 - 2.0 / M_PI * std::atan(dist / scale);
        }

        Eigen::MatrixXd compute_dynamic_target(const Eigen::VectorXd & target, int timestep) {
            Eigen::MatrixXd dynamic_target = target; // copy
            double duration = (timestep - target(timestep_idx(),0)) * dt();
            dynamic_target(0) = dynamic_target(0) + target(3,0) * duration; 
            dynamic_target(1) = dynamic_target(1) + target(4,0) * duration; 
            dynamic_target(2) = dynamic_target(2) + target(5,0) * duration; 
            return dynamic_target;
        }

        double V(Eigen::VectorXd state) override {
            return m_sixdof->V(state); }

        double V(Eigen::VectorXd state, RNG& rng) override {
            return m_sixdof->V(state, rng); }

        int H() override { return m_sixdof->H(); }

        double gamma() override { return m_sixdof->gamma(); }

        Eigen::Matrix<double,6,1> aero_model(const Eigen::VectorXd & state, const Eigen::VectorXd & action) {
            return m_sixdof->aero_model(state.head(not_augmented_state_dim()), action); }

        // jacobian information for SCP
        Eigen::MatrixXd dFdx(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return F(x, action);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(state, f, fjac, accuracy, 1.0e-5);
            return fjac; 
        }

        Eigen::MatrixXd dFdu(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> Eigen::VectorXd {
                return F(state, u);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(action, f, fjac, accuracy, 1.0e-5);
            return fjac; 
        }

        Eigen::MatrixXd dFdx_non_augmented(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return m_sixdof->F(x, action);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(state, f, fjac, accuracy, 1.0e-5);
            return fjac; 
        }

        Eigen::MatrixXd dFdu_non_augmented(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> Eigen::VectorXd {
                return m_sixdof->F(state, u);
            };
            fd::AccuracyOrder accuracy = fd::SECOND;
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(action, f, fjac, accuracy, 1.0e-5);
            return fjac; 
        }

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

        Eigen::VectorXd get_xd() override { 
            int xd_size = 0;
            for (int ii=0; ii<m_targets.size(); ii++) {
                xd_size += m_targets[ii].size();
            }
            Eigen::VectorXd xd(xd_size);
            int idx = 0;
            for (int ii=0; ii<m_targets.size(); ii++) {
                xd.segment(idx, m_targets[ii].size()) = m_targets[ii];
                idx += m_targets[ii].size();
            }
            return xd;
        }

        Eigen::Matrix<double,6,1> compute_aero_forces_and_moments_from_state_diff(
                Eigen::Matrix<double,13,1> actual_state, Eigen::Matrix<double,13,1> predicted_state, double dt) {
            return m_sixdof->compute_aero_forces_and_moments_from_state_diff(actual_state, predicted_state, dt);
        }

        
    private: 
        SixDOFAircraft* m_sixdof;
        std::vector<Eigen::VectorXd> m_targets;
        std::vector<double> m_reward_parameters;
        std::vector<int> m_special_target_idxs;
};
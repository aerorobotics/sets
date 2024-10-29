#pragma once

#include <algorithm>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <limits>
#include <queue>
#include <numeric>
#include <chrono>

#include "solver.hpp"


class UD_PS : public Solver {

    public: 

        UD_PS() { };


        void set_param(int N, int max_depth, double wct, double c, bool export_topology, 
                bool export_states, bool export_trajs, bool export_cbdsbds, bool export_tree_statistics, 
                std::string heuristic_mode, std::string tree_exploration, bool downsample_trajs, 
                int num_points_per_dimension, int num_timesteps_hold, bool verbose) {

            m_N = N; 
            m_max_depth = max_depth; 
            m_wct = wct; 
            m_c = c; 
            m_export_topology = export_topology;
            m_export_node_states = export_states;
            m_export_trajs = export_trajs;
            m_export_cbdsbds = export_cbdsbds;
            m_export_tree_statistics = export_tree_statistics;
            m_heuristic = heuristic_mode;
            m_tree_exploration = tree_exploration;
            m_downsample_trajs = downsample_trajs;
            m_num_points_per_dimension = num_points_per_dimension;
            m_num_timesteps_hold = num_timesteps_hold;
            m_verbose = verbose;
        
        }


        struct Node {
            double max_value = 0.0;
            double total_value = 0.0;
            int num_visits = 0;
            int idx; 
            int parent_idx; 
            int branch_idx; 
            int depth; 
            int time_of_expansion; 
            std::vector<std::shared_ptr<Node>> children;
            Trajectory traj; // traj to node. Node is valid if traj is valid 
        };


        void print_node(Node* node) {
            std::cout << "node->idx: " << node->idx << std::endl;
            std::cout << "node->parent_idx: " << node->parent_idx << std::endl;
            std::cout << "node->num_visits: " << node->num_visits << std::endl;
            std::cout << "node->max_value: " << node->max_value << std::endl;
            std::cout << "node->total_value: " << node->total_value << std::endl;
            std::cout << "node->children.size(): " << node->children.size() << std::endl;
            std::cout << "node->branch_idx: " << node->branch_idx << std::endl;
            std::cout << "node->time_of_expansion: " << node->time_of_expansion << std::endl;
            std::cout << "node->depth: " << node->depth << std::endl;
            std::cout << "node->traj.xs.back(): "; print_v(node->traj.xs.back());
        }


        int random_child(RNG& rng, Node* node, MDP* mdp) {
            // if node hasnt been expanded, return that one immediately            
            // if all nodes are expanded and there exists a valid node, return the best value valid node
            // if all nodes are expanded and none are valid, return -1

            int argmax = -1;
            double best_val = -1.0;

            std::vector<int> perm(node->children.size()); 
            perm = rng.permutation(node->children.size());

            for (int ii : perm) {
                if (node->children[ii] == nullptr) {
                    argmax = ii;
                    break; 
                } else if (node->children[ii]->traj.is_valid) {
                    argmax = ii; 
                    break;     
                }                    
            }
            
            return argmax; 
        }


        int max_ave_child(RNG& rng, Node* node, MDP* mdp) {
            int argmax = -1;
            double best_val = -1.0;
            std::vector<int> perm = rng.permutation(node->children.size());
            for (int ii : perm) {
                if (node->children[ii] != nullptr) {
                    double curr_val = node->children[ii]->total_value / node->children[ii]->num_visits;
                    if (curr_val > best_val) {
                        argmax = ii;
                        best_val = curr_val; 
                    } 
                } 
            }
            return argmax; 
        }


        void export_node_states(Tree& tree) {
            // tree.node_states.resize(m_nodes.size());
            for (int ii=0; ii<m_nodes.size(); ii++) {
                if (m_nodes[ii]->traj.xs.size() > 0) {
                    tree.node_states.push_back(m_nodes[ii]->traj.xs.back());
                }
            }
        }


        void export_tree_topology(Tree& tree) {
            tree.topology.resize(m_nodes.size(),8);
            for (int ii=0; ii<m_nodes.size(); ii++) {
                tree.topology(ii,0) = m_nodes[ii]->idx;
                tree.topology(ii,1) = m_nodes[ii]->parent_idx;
                tree.topology(ii,2) = m_nodes[ii]->num_visits;
                tree.topology(ii,3) = m_nodes[ii]->max_value;
                tree.topology(ii,4) = m_nodes[ii]->traj.is_valid;
                tree.topology(ii,5) = m_nodes[ii]->branch_idx;
                tree.topology(ii,6) = m_nodes[ii]->depth;
                tree.topology(ii,7) = m_nodes[ii]->time_of_expansion;
            } 
        }


        void export_tree_statistics(Tree& tree) {
            tree.node_visit_statistics.resize(m_nodes.size());
            for (int ii=0; ii<m_nodes.size(); ii++) {
                tree.node_visit_statistics[ii] = std::make_tuple(
                    m_nodes[ii]->max_value, m_nodes[ii]->num_visits, m_nodes[ii]->depth);
            } 
        }


        // rolls out from node, returns value of trajectory 
        double rollout(RNG& rng, int rollout_count, Node* node, int depth, Trajectory &traj_from_root, MDP *mdp, Tree & tree) {

            if (m_verbose) { std::cout << "depth: " << depth << std::endl; }
            if (m_verbose) { std::cout << "node: "; print_node(node); }

            // terminal 
            if (depth >= m_max_depth) {
                if (m_verbose) { std::cout << "reached terminal depth." << std::endl; }
                return mdp->V(node->traj.xs.back(), rng);
            }

            // best child looks until valid expansion 
            int argmax; 
            while (true) {

                // find best 
                if (m_verbose) { std::cout << "max_uct_child..." << std::endl; }
                argmax = random_child(rng, node, mdp);

                // if argmax is -1, all children are invalid 
                if (argmax == -1) { 
                    if (m_verbose) { 
                        std::cout << "no possible expansion without violating constraints." << std::endl; 
                        std::cout << "x: "; print_v(node->traj.xs.back()); }
                    return 0.0; 
                }

                // if node does not exist, expand node 
                if (node->children[argmax] == nullptr) {

                    // alloc memory and initialize node
                    node->children[argmax] = std::shared_ptr<Node>( new Node );
                    node->children[argmax]->idx = m_nodes.size();
                    node->children[argmax]->parent_idx = node->idx;
                    node->children[argmax]->branch_idx = argmax;
                    node->children[argmax]->depth = depth+1;
                    node->children[argmax]->time_of_expansion = rollout_count;
                    node->children[argmax]->children.resize(m_actionSet_eig.size());
                    m_nodes.push_back(node->children[argmax]);

                    // expand, writes to node->children[argmax]->traj
                    if (m_verbose) { std::cout << "expand node..." << std::endl; }

                    Eigen::VectorXd u = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(m_actionSet_eig[argmax].data(), m_actionSet_eig[argmax].size());
                    std::vector<Eigen::VectorXd> us(m_num_timesteps_hold, u);
                    node->children[argmax]->traj = rollout_action_sequence(node->traj.xs.back(), us, mdp, true);

                    if (m_downsample_trajs) {
                        if (m_verbose) {
                            std::cout << "   regular traj: " << std::endl;
                            print_traj(node->children[argmax]->traj);
                        } 
                        if (node->children[argmax]->traj.xs.size()>2) {
                            node->children[argmax]->traj.xs.erase(node->children[argmax]->traj.xs.begin(), node->children[argmax]->traj.xs.end()-1);
                            node->children[argmax]->traj.us.erase(node->children[argmax]->traj.us.begin(), node->children[argmax]->traj.us.end()-1);
                            node->children[argmax]->traj.rs.erase(node->children[argmax]->traj.rs.begin(), node->children[argmax]->traj.rs.end()-1);
                        }
                        if (m_verbose) {
                            std::cout << "   downsampled traj: " << std::endl;
                            print_traj(node->children[argmax]->traj);
                        }
                    }

                }

                // if node is valid, break 
                if (node->children[argmax]->traj.is_valid) { 
                    break; 
                } 
                else {
                    return 0.0;
                }
            }


            extend_traj(traj_from_root, node->children[argmax]->traj);
            double reward = node->children[argmax]->traj.value;
            if (m_verbose) { std::cout << "node->children[argmax]->traj.value: " << node->children[argmax]->traj.value << std::endl; }
            if (m_verbose) { std::cout << "rollout depth: " << depth << " completed" << std::endl; }
            double value = rollout(rng, rollout_count, node->children[argmax].get(), depth+1, traj_from_root, mdp, tree);
            
            // backup
            node->children[argmax]->total_value += value + reward;
            node->children[argmax]->num_visits += 1;
            node->children[argmax]->max_value = std::max(value + reward, node->max_value);
            if (m_verbose) { std::cout << "backup depth: " << depth << " completed" << std::endl; }
            return value + reward;
        }


        SolverResult solve(Eigen::VectorXd state, int timestep, MDP *mdp, RNG& rng) override {

            if (m_verbose) { std::cout << "solve..." << std::endl; }

            m_actionSet_eig = makeActionSet(mdp->U(), mdp->action_dim(), m_num_points_per_dimension);

            // get start time 
            std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();

            // internal data structures 
            m_nodes.resize(0);

            // root 
            std::shared_ptr<Node> root(new Node);
            root->traj.xs = std::vector<Eigen::VectorXd> (1,state);
            root->traj.us = std::vector<Eigen::VectorXd> (1,mdp->empty_control());
            root->idx = 0;
            root->parent_idx = -1;
            root->branch_idx = -1;
            root->depth = 0;
            root->time_of_expansion = 0;
            root->children.resize(m_actionSet_eig.size());
            m_nodes.push_back(root);

            // external datastructures 
            Tree tree;
            tree.root = state;
            Trajectory max_max_traj; 
            double max_max_value = -1.0;
            std::vector<double> max_max_values;

            // compute
            for (int ii = 0; ii < m_N; ii++) {

                if (std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now() - start_time).count() > 1000 * m_wct) {
                    break; 
                }
                
                // rollout
                if (m_verbose) { std::cout << "rollout ii: " << ii << std::endl; }
                // if (true) { std::cout << "rollout ii: " << ii << std::endl; }

                Trajectory curr_traj; 
                double value = rollout(rng, ii, root.get(), 0, curr_traj, mdp, tree);
                root->num_visits += 1; // last backup

                if (m_export_trajs) {
                    tree.trajs.push_back(curr_traj);
                }

                // update internal states
                if (value > max_max_value) { 
                    max_max_traj = curr_traj; 
                    max_max_value = value; 
                }
                max_max_values.push_back(max_max_value);
            }

            // export_node_states
            if (m_verbose) { std::cout << "export_tree" << std::endl; }
            if (m_export_topology) { 
                export_tree_topology(tree); 
            }

            if (m_export_node_states) {
                export_node_states(tree); 
            }

            if (m_export_tree_statistics) {
                export_tree_statistics(tree);
            }

            // get max_average action (could be different from max_max trajectory)
            if (m_verbose) { std::cout << "max_ave_child" << std::endl; }
            // int argmax = max_ave_child(rng, root, mdp);
            int argmax = max_ave_child(rng, root.get(), mdp);
            
            // couldnt expand any nodes from root without violating constraints 
            if (argmax == -1) {
                std::cout << "no possible expansions without violating constraints." << std::endl; 
                std::cout << "x: "; print_v(state); 
                SolverResult r;
                r.success = false; 
                return r; }

            // else, "success"
            if (m_verbose) { std::cout << "max_ave_u" << std::endl; }
            Trajectory max_ave_traj = root->children[argmax]->traj;
            
            // Fill the essential parts of the output struct.
            SolverResult r;
            r.success = true;
            r.planned_traj = max_max_traj;
            r.mpc_traj = max_ave_traj;
            r.vs = max_max_values;
            r.tree = tree;

            if (m_verbose) { std::cout << "solve done!" << std::endl; }
            return r;
        }


    private:
        int m_N; 
        int m_max_depth; 
        double m_c; 
        double m_wct; 
        bool m_export_topology;
        bool m_export_node_states;
        bool m_export_trajs;
        bool m_export_cbdsbds;
        bool m_export_tree_statistics;
        bool m_verbose;
        bool m_downsample_trajs;
        std::string m_heuristic;
        std::string m_tree_exploration;
        std::vector<std::shared_ptr<Node>> m_nodes;
        std::vector<Eigen::VectorXd> m_actionSet_eig; 
        int m_num_points_per_dimension;
        int m_num_timesteps_hold;
};

#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>

#include "../util/util.hpp"


class MDP {

    // M = <X, U, F, R, H, gamma>
    // - state vector must always include timestep

    public:

        virtual std::string name() {
            throw std::logic_error("name not implemented"); 
        }

        virtual void expand_ii(RNG& rng, int branch_idx, const Eigen::VectorXd & prev_state, 
            const Eigen::VectorXd & prev_control, Trajectory & traj, CommonBranchData & cbd, 
            SpecificBranchData & sbd, WallClockTimeData & wctdata) {
            throw std::logic_error("expand_ii not implemented"); 
        }

        virtual void expand_ii_mem_safe(RNG& rng, int branch_idx, const Eigen::VectorXd & prev_state, 
            const Eigen::VectorXd & prev_control, Trajectory & traj) {
            throw std::logic_error("expand_ii not implemented"); 
        }

        virtual std::vector<int> velocity_idxs() {
            throw std::logic_error("velocity_idxs not implemented"); 
        }

        virtual std::vector<int> position_idxs() {
            throw std::logic_error("position_idxs not implemented"); 
        }

        virtual std::vector<int> my_idxs() {
            throw std::logic_error("my_idxs not implemented"); 
        }

        virtual int state_dim() { 
            throw std::logic_error("state_dim not implemented"); }

        virtual int not_augmented_state_dim() { 
            throw std::logic_error("not_augmented_state_dim not implemented"); }

        virtual int action_dim() { 
            throw std::logic_error("control_dim not implemented"); }

        virtual double dt() { 
            throw std::logic_error("dt not implemented"); }

        virtual int timestep_idx() {
            throw std::logic_error("timestep_idx not implemented"); } 
        
        virtual int num_actions() { 
            throw std::logic_error("num_actions not implemented"); }

        virtual bool is_state_valid(const Eigen::VectorXd & state) { 
            throw std::logic_error("is_state_valid not implemented"); }

        virtual Eigen::VectorXd initial_state() { 
            throw std::logic_error("initial_state not implemented"); }

        virtual Eigen::VectorXd empty_control() { 
            throw std::logic_error("empty_control not implemented"); }

        virtual void set_xd(Eigen::VectorXd _xd) {
            throw std::logic_error("set_xd not implemented"); }

        virtual void set_x0(Eigen::VectorXd _x0) {
            throw std::logic_error("set_x0 not implemented"); }

        // The state space is a hypercube.
        virtual Eigen::Matrix<double,-1,2> X() { 
            throw std::logic_error("X not implemented"); }

        // The action space is a hypercube. 
        virtual Eigen::Matrix<double,-1,2> U() { 
            throw std::logic_error("U not implemented"); }

        // The discrete action space is a discrete set of sequence of vectors.
        virtual std::vector<std::vector<Eigen::VectorXd>> U_d(const Eigen::VectorXd & state, RNG& rng) { 
            throw std::logic_error("U_d not implemented"); }

        virtual Eigen::VectorXd F(const Eigen::VectorXd & state, const Eigen::VectorXd & action) { 
            throw std::logic_error("F not implemented"); 
        }

        virtual Eigen::MatrixXd B(const Eigen::VectorXd & state) { 
            throw std::logic_error("B not implemented"); 
        }

        virtual Eigen::VectorXd F_non_augmented(const Eigen::VectorXd & state, const Eigen::VectorXd & action) { 
            throw std::logic_error("F_non_augmented not implemented"); 
        }            

        Eigen::VectorXd F_timeless(const Eigen::VectorXd & state_timeless, const Eigen::VectorXd & action) { 
            // state : Eigen::Vector in (m_state_dim-1) (i.e. timeless)
            Eigen::VectorXd state = Eigen::VectorXd::Zero(this->state_dim());
            state.block(0,0,this->state_dim()-1,1) = state_timeless;
            Eigen::VectorXd next_state = this->F(state, action); 
            removeRow(next_state, this->timestep_idx());
            return next_state;
        }

        virtual double R(const Eigen::VectorXd & state, const Eigen::VectorXd & action) { 
            throw std::logic_error("R not implemented"); }

        virtual double R_verbose(const Eigen::VectorXd & state, const Eigen::VectorXd & action, bool verbose) { 
            throw std::logic_error("R_verbose not implemented"); }

        virtual int H() { 
            throw std::logic_error("H not implemented"); }
        
        virtual double gamma() { 
            throw std::logic_error("gamma not implemented"); }

        virtual std::vector<Eigen::MatrixXd> obstacles() {
            return m_obstacles; }

        // used for real-time environment updates
        virtual void add_obstacle(Eigen::Matrix<double,-1,2> obstacle) { 
            throw std::logic_error("add_obstacle not implemented"); }

        virtual void clear_obstacles() { 
            throw std::logic_error("clear_obstacles not implemented"); }

        virtual void add_target(Eigen::VectorXd target) { 
            throw std::logic_error("add_target not implemented"); }

        virtual void clear_targets() { 
            throw std::logic_error("clear_targets not implemented"); }

        virtual void add_thermal(Eigen::MatrixXd X_thermal, Eigen::VectorXd V_thermal) { 
            throw std::logic_error("add_thermal not implemented"); }

        virtual void clear_thermals() { 
            throw std::logic_error("clear_obstacles not implemented"); }

        virtual Eigen::VectorXd update_augmented_state_only(const Eigen::VectorXd & state, const Eigen::VectorXd & action) { 
            throw std::logic_error("update_augmented_state_only not implemented"); }

        // only used for dispersion 
        virtual void add_traj(Trajectory traj) { 
            throw std::logic_error("add_traj not implemented"); }

        virtual void set_trajs(std::vector<Trajectory> trajs) { 
            throw std::logic_error("set_trajs not implemented"); }

        Eigen::VectorXd sample_state(RNG& rng) {
            Eigen::VectorXd state(this->state_dim());
            for (int ii=0; ii < this->state_dim(); ii++) {
                state(ii) = (this->X()(ii,1) - this->X()(ii,0)) * rng.uniform() + this->X()(ii,0); }
            // always sample at timestep 0
            state(this->timestep_idx(),0) = 0;
            return state; }
        
        Eigen::VectorXd sample_action(RNG& rng) {
            Eigen::VectorXd action(this->action_dim());
            for (int ii=0; ii < this->action_dim(); ii++) {
                action(ii) = (this->U()(ii,1) - this->U()(ii,0)) * rng.uniform() + this->U()(ii,0); }
            return action; }

        // only used for setting nn weights 
        virtual void set_weights(std::vector<Eigen::MatrixXd> weightss, std::vector<Eigen::MatrixXd> biass) { 
            throw std::logic_error("set_weights not implemented"); }

        // for testing purposes
        virtual Eigen::VectorXd eval_ff(const Eigen::VectorXd & state, const Eigen::VectorXd & action) { 
            throw std::logic_error("eval_ff not implemented"); }

        // leaf oracle 
        virtual double V(Eigen::VectorXd state) { 
            return 0.0; }

        virtual double V(Eigen::VectorXd state, RNG& rng) {
            return 0.0; }

        // only used for SCP
        virtual Eigen::VectorXd dVdx(Eigen::VectorXd state) { 
            throw std::logic_error("dVdx not implemented"); }

        virtual Eigen::MatrixXd d2Vdx2(Eigen::VectorXd state) { 
            throw std::logic_error("d2Vdx2 not implemented"); }

        virtual Eigen::MatrixXd dFdz(Eigen::VectorXd stateaction) {
            throw std::logic_error("dFdz not implemented"); }

        virtual Eigen::MatrixXd dFdx(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("dFdx not implemented"); }

        virtual Eigen::MatrixXd dFdu(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("dFdu not implemented"); }

        virtual Eigen::MatrixXd dFdx_non_augmented(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("dFdx_non_augmented not implemented"); }

        virtual Eigen::MatrixXd dFdu_non_augmented(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("dFdu_non_augmented not implemented"); }

        virtual Eigen::VectorXd dRdx(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("dRdx not implemented"); }

        virtual Eigen::VectorXd dRdu(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("dRdu not implemented"); }

        virtual Eigen::MatrixXd d2Rdx2(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("d2Rdx2 not implemented"); }

        virtual Eigen::MatrixXd d2Rdx2_inv(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("d2Rdx2_inv not implemented"); }

        virtual Eigen::MatrixXd d2Rdu2(Eigen::VectorXd state, Eigen::VectorXd action) {
            throw std::logic_error("d2Rdu2 not implemented"); }

        virtual Eigen::VectorXd get_xd() {
            throw std::logic_error("get_xd not implemented"); 
        }

        virtual void set_dt(double dt) {
            throw std::logic_error("set_dt not implemented"); 
        }

        virtual Eigen::MatrixXd sqrtQx() {
            throw std::logic_error("sqrtQx not implemented"); }

        virtual Eigen::MatrixXd sqrtQx_equ() {
            throw std::logic_error("sqrtQx not implemented"); }

        virtual Eigen::MatrixXd sqrtQu() {
            throw std::logic_error("sqrtQu not implemented"); }

        virtual Eigen::MatrixXd sqrtQf() {
            throw std::logic_error("sqrtQf not implemented"); }

    protected:
        std::vector<Eigen::MatrixXd> m_obstacles;

};


Trajectory rollout_action_sequence(Eigen::VectorXd curr_state, 
                                   std::vector<Eigen::VectorXd> action_sequence, 
                                   MDP* mdp,
                                   bool break_when_invalid) {
    Trajectory traj; 
    bool is_valid = true;
    for (Eigen::VectorXd u : action_sequence) {
        curr_state = mdp->F(curr_state, u);
        double r = mdp->R(curr_state, u);
        is_valid = is_valid && mdp->is_state_valid(curr_state);
        traj.xs.push_back(curr_state);
        traj.us.push_back(u);
        traj.rs.push_back(r); 
        if (break_when_invalid && !is_valid) { break; }
    }
    // todo: account for discount in sum 
    traj.value = std::accumulate(traj.rs.begin(), traj.rs.end(), 0.0) / traj.xs.size();
    // traj.value = std::accumulate(traj.rs.begin(), traj.rs.end(), 0.0);
    traj.is_valid = is_valid;
    return traj; 
}

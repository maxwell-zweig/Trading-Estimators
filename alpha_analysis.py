import numpy as np
import cmath
import math 
import time
import numba as nb 
import matplotlib.pyplot as plt 
import scipy.optimize as optim
import matplotlib.pyplot as plt
from sympy import symbols, Eq
from sympy.solvers import solve 
import torch


class Simulation:
    def __init__(self, dt):
        self.b = 7
        self.theta = 100
        self.n = 3
        self.v = 2
        self.alpha_mean = 0 
        self.mid_sigma = 40
        self.dt = dt

        self.k = 0.2
        self.zeta = 10
        self.sigma_alpha = 30
        self.epsilon = 5

        self.buy_rate = self.theta
        self.sell_rate = self.theta 
        self.drift = 0

        self.current_time = 0 
        self.current_price = 1000


        self.buy_rates = []
        self.sell_rates = []
        self.prices = []



    def update_drift(self, received_buy, received_sell):
        direction = np.random.randint(0, 2)
        if direction == 0:
            self.drift += -1 * self.zeta * self.drift * self.dt - self.sigma_alpha * math.sqrt(self.dt) + self.epsilon * received_buy - self.epsilon * received_sell
        else:
            self.drift += -1 * self.zeta * self.drift * self.dt + self.sigma_alpha * math.sqrt(self.dt) + self.epsilon * received_buy - self.epsilon * received_sell


    def update_orderflow(self, received_buy, received_sell):
        """
        Updates the orderflow state parameters. 
        
        Parameter received_buy: 1 if a buy was received, 0 otherwise. 
        
        Parameter received_sell: 1 if a sell was received, 0 otherwise.
        """
        self.buy_rate += self.b * (self.theta - self.buy_rate) * self.dt + self.n * received_buy + self.v * received_sell
        self.sell_rate += self.b * (self.theta - self.sell_rate) * self.dt + self.n * received_sell + self.v * received_buy

    def update_price(self):
        direction = np.random.randint(0, 2)
        if direction == 0:
            self.current_price += -1 * self.mid_sigma * math.sqrt(self.dt) + (self.alpha_mean + self.drift) * self.dt 
        else:
            self.current_price += self.mid_sigma * math.sqrt(self.dt) + (self.alpha_mean + self.drift) * self.dt 

         

    def order_filled(self, price, side):
        if side == "buy":
            delta = self.current_price - price
            rate = self.buy_rate
        else:
            delta = price - self.current_price
            rate = self.sell_rate
        if delta < 0:
            intensity = rate * self.dt
            hit = 0 == np.random.randint(0, int((1 / intensity)))
        else:
            intensity = rate * np.exp(-self.k * delta) * self.dt
            if intensity == 0 or math.isinf(1 / intensity) or int(1 / intensity) >= np.iinfo(np.int32).max:
                hit = False
            elif int((1 / intensity)) == 0:
                hit = True
            else:
                hit = 0 == np.random.randint(0, int((1 / intensity))) 
        return hit 

    def fit(self, dt):

        @nb.jit(nopython=True)
        def generate_fitting_data(dt):
            states = []
            order_rec = []
            buy_lambdas = []
            sell_lambdas = []
            mts = []
            drift = 0 
            current_time = 0 


            b = 500
            theta = 100
            n = 400
            v = 70
            alpha_mean = 0 
            mid_sigma = 40

            zeta = 100
            sigma_alpha = 0.01
            epsilon = 0.05
            lt = 0


           # print(f'Zeta Target: {zeta} Sigma Target: {sigma_alpha} Epsilon Target: {epsilon}, Theta Target: {lt}')

            print(zeta, sigma_alpha, lt, epsilon)

            buy_rate = theta
            sell_rate = theta 
            drift = lt

            while current_time <= 1:
                
                buy_intensity = dt * buy_rate
                rec_buy = 0 == np.random.randint(0, int((1 / buy_intensity))) 

                sell_intensity = dt * sell_rate
                rec_sell = 0 == np.random.randint(0, int((1 / sell_intensity))) 

                while rec_buy and rec_sell:
                    buy_intensity = dt * buy_rate
                    rec_buy = 0 == np.random.randint(0, int((1 / buy_intensity))) 

                    sell_intensity = dt * sell_rate
                    rec_sell = 0 == np.random.randint(0, int((1 / sell_intensity))) 

                rec_buy_int = 1 if rec_buy else 0 
                rec_sell_int = 1 if rec_sell else 0 


                if rec_buy and not rec_sell:
                    order_rec.append(1)
                elif rec_sell and not rec_buy:
                    order_rec.append(-1)
                elif not rec_sell and not rec_buy:
                    order_rec.append(0)
                elif rec_sell and rec_buy:
                    print("bruh")
                states.append(drift)
                mts.append(current_time)

                buy_rate += b * (theta - buy_rate) * dt + n * rec_buy_int + v * rec_sell_int
                sell_rate += b * (theta - sell_rate) * dt + n * rec_sell_int + v * rec_buy_int

                direction = np.random.randint(0, 2)
                if direction == 0:
                    drift += zeta * (lt - drift) * dt - sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                else:
                    drift += zeta * (lt - drift) * dt + sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                current_time += dt 
                buy_lambdas.append(buy_rate)
                sell_lambdas.append(sell_rate)
                

            return states, order_rec, buy_lambdas, sell_lambdas, mts
        states, order_rec, bl, sl, mt = generate_fitting_data(dt)
        plt.plot(states)
        plt.show()
        return np.array(states), np.array(order_rec), np.array(bl), np.array(sl), np.array(mt)



class MLE:
    def __init__(self, events, states, measurement_times):
      #  self.dt = dt
        self.events = events
        self.states = states
        self.measurement_times = measurement_times
        self.delta_times = measurement_times[1:] - measurement_times[:-1]
        self.delta_times = self.delta_times
        


        self.delta_states = states[1:] - states[:-1]
        self.cur_states = states[:-1]
        self.delta_events = self.events[:-1]


        self.delta_states = self.delta_states
        self.delta_events = self.delta_events
        self.cur_states = self.cur_states

        self.num_samples = self.delta_states.shape[0]
        self.num_sells = self.delta_events[self.delta_events == -1].shape[0]
        self.num_buys = self.delta_events[self.delta_events == 1].shape[0]


        self.dstatesallsquared = np.sum(self.delta_states ** 2)
        self.dstatescurstates = np.sum(np.multiply(self.cur_states, self.delta_states))
        self.dstatesall = np.sum(self.delta_states)
        self.statesall = np.sum(self.cur_states)
        self.statesallsquared = np.sum(self.cur_states ** 2)

        self.dstatessell = self.delta_states[self.delta_events == -1]
        self.cur_statessell = self.cur_states[self.delta_events == -1]
        self.dtimessell = self.delta_times[self.delta_events == -1]

        self.dstatesbuy = self.delta_states[self.delta_events == 1]
        self.cur_statesbuy = self.cur_states[self.delta_events == 1]
        self.dtimesbuy = self.delta_times[self.delta_events == 1]

        self.dstatesnone = self.delta_states[self.delta_events == 0]
        self.cur_statesnone = self.cur_states[self.delta_events == 0]
        self.dtimesnone = self.delta_times[self.delta_events == 0]

        self.dstatessell_sm = np.sum(self.delta_states[self.delta_events == -1])
        self.cur_statessell_sm = np.sum(self.cur_states[self.delta_events == -1])

        self.dstatesbuy_sm = np.sum(self.delta_states[self.delta_events == 1])
        self.cur_statesbuy_sm = np.sum(self.cur_states[self.delta_events == 1])

        self.dstatesnone_sm = np.sum(self.delta_states[self.delta_events == 0])
        self.cur_statesnone_sm = np.sum(self.cur_states[self.delta_events == 0])


    def maxim(self):
        k, variance, lt_f,  epsilon = self.closed_form_mt_alteration(dstatesall_sm_=self.dstatesall, 
                                                                    statesall_wsm=np.sum(self.cur_states * self.delta_times),
                                                                    num_buys=self.num_buys,
                                                                    num_sells= self.num_sells, 
                                                                    drdtB_= np.sum(self.dstatesbuy / self.delta_times[self.delta_events == 1]),
                                                                    rsmB_=self.cur_statesbuy_sm,
                                                                    dt1B_= np.sum(1 / self.delta_times[self.delta_events == 1]),
                                                                    drdtS_=np.sum(self.dstatessell / self.delta_times[self.delta_events == -1]),
                                                                    rsmS_=self.cur_statessell_sm,
                                                                    dt1s_= np.sum(1 / self.delta_times[self.delta_events == -1]),
                                                                    num_msm_=self.num_samples,
                                                                    dstatesallstates_sm_=self.dstatescurstates,
                                                                    states_all_sm_=self.statesall,
                                                                    statesall2_wsm_= np.sum(self.cur_states * self.cur_states * self.delta_times), 
                                                                    total_time = self.measurement_times[-1] - self.measurement_times[0])
                                                                
#        print(k, lt_f, variance, epsilon)

        return k, variance, lt_f, epsilon

    '''
    
    def closed_form(self, dt, dstatesallsquared, dstatesall, statesallsquared, statesall, dstatescurstates, num_samples, dstatesbuy, curstatesbuy, num_buys, dstatessell, curstatessell, num_sells):
        
        
        theta, N1, N3, N2, C6, C7, K, DT, C4, C5, C0, C1, C2, C3 = symbols('theta, N1, N3, N2, C6, C7, K, DT, C4, C5, C0, C1, C2, C3')
        final_eq = Eq(((-C4 + C5) / (N1 + N3)) * (C6 - C7 + K * DT * (C4 - C5) + K * theta * DT * (-N1 + N3) ) -  theta * K * DT * C1 + C2 + K * DT * C3, 0)

        final_eq_subs = final_eq.subs(theta, ((((N3 - N1) / (N1 + N3)) * (C6 - C7 + K * DT * (C4 - C5)) - C0 - K * DT * C1)  / (K * DT * (-N2 - ((N1 - N3) / (N1 + N3))  * (N3 - N1)))))

        substitutions = [(N1, num_sells), (N3, num_buys), (N2, num_samples), (C6, dstatesbuy), (C7, dstatessell), (DT, dt), (C4, curstatesbuy), (C5, curstatessell), (C0, dstatesall), (C1, statesall), (C2, dstatescurstates), (C3, statesallsquared)]

        k_solved = solve(final_eq_subs, K)

        final_0 = k_solved[0].subs(substitutions)

        k_est = final_0

        imb = (num_sells - num_buys) / (num_sells + num_buys)
        theta = lambda k : (imb * (dstatesbuy - dstatessell + k * dt *(curstatesbuy - curstatessell)) - dstatesall - k*dt * statesall) /(k * dt * (-num_samples - imb * (num_buys - num_sells)))
        eps = lambda k, th : (1 / (num_sells + num_buys)) * (dstatesbuy - dstatessell + k*dt*(curstatesbuy - curstatessell) + k * th * dt *(num_sells - num_buys))


        full_sample = lambda k, th:  dstatesallsquared - 2 * k * th * dt * dstatesall + 2 * k * dt * dstatescurstates - 2 * (k ** 2) * th * (dt ** 2) * statesall + (k ** 2) * (dt ** 2) * statesallsquared + num_samples * (th ** 2) * (k ** 2) * (dt ** 2)
        buys = lambda k, th, epsi: -2 * epsi * dstatesbuy - 2 * epsi * k * dt  * curstatesbuy + 2 * epsi * k * th * dt * num_buys + (epsi ** 2) * num_buys
        sells = lambda k, th, epsi: 2 * epsi * dstatessell + 2 *  epsi * k * dt  * curstatessell - 2 * epsi * k * th * dt * num_sells + num_sells * (epsi ** 2)
        
        lt = theta(k_est)
        epsilon = eps(k_est, lt)

        var = lambda k, th, eps : (full_sample(k, th) + buys(k, th, eps) + sells(k, th, eps)) / (num_samples * dt)

        variance = var(k_est, lt, epsilon)

        return k_est, lt, variance, epsilon
        
    def computational_maximization(self):
        def loss(x, dt, dstatesbuy, curstatesbuy, dstatessell, curstatessell, dstatesnone, curstatesnone):
            k, var, theta, eps = x[0], x[1], x[2], x[3]

            num_buys = dstatesbuy.shape[0]
            num_sells = dstatessell.shape[0]
            num_none = dstatesnone.shape[0]

            buy_portion = -1 / (2 * var ** 2 * dt) * np.sum((dstatesbuy - k * (theta - curstatesbuy) * dt - eps) ** 2) - num_buys * 0.5 * np.log(2 * math.pi * var ** 2 * dt)
            sell_portion = -1 / (2 * var ** 2 * dt) * np.sum((dstatessell - k *(theta - curstatessell) * dt + eps) ** 2) - num_sells * 0.5 * np.log(2 * math.pi * var ** 2 * dt)
            none_portion = -1 / (2 * var ** 2 * dt) * np.sum((dstatesnone - k *(theta - curstatesnone) * dt) ** 2) - num_none * 0.5 * np.log(2 * math.pi * var ** 2 * dt)
            return -1 * (buy_portion + none_portion + sell_portion)
        
        initial_theta_estimate = np.mean(self.cur_states)
        eps_bound = (np.max(self.cur_states) - np.min(self.cur_states)) / 2
        theta_bound = np.max(self.cur_states)

       # print(x0)cd
        x0 = [0.1,0.1,0.1,0.1]

        args_tuple = (self.dt, self.dstatesbuy, self.cur_statesbuy, self.dstatessell, self.cur_statessell, self.dstatesnone, self.cur_statesnone)
        
        bound = optim.Bounds((0, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf))
        
       # res = optim.minimize(loss, x0=x0, args=args_tuple, method="L-BFGS-B", options={'maxiter' : 10000}, bounds=bound)
        print('starting')
        res = optim.brute(loss, ranges=((18, 21), (0.01, 0.1), (-0.1, 0.1), (-0.1, 0.1)), args=args_tuple, Ns=2)
        print('done')

        print(res)

        return res.x
        '''
    def omega_calc(self, k, theta, eps, dstatesbuy, curstatesbuy, dstatessell, curstatessell, dstatesnone, curstatesnone, dtimesnone, dtimesbuy, dtimessell):
        

        k, theta, eps = float(k), float(theta), float(eps)

      #  print('doubly stukc')
        no_trade_portion =  np.sum((dstatesnone - k * (theta - curstatesnone) * dtimesnone) ** 2 / dtimesnone) 
        buy_portion =  np.sum((dstatesbuy - k * (theta - curstatesbuy) * dtimesbuy - eps) ** 2 / dtimesbuy) 
        sell_portion =  np.sum((dstatessell - k * (theta - curstatessell) * dtimessell + eps) ** 2 / dtimessell) 
      #  print('bruh')
        return no_trade_portion + buy_portion + sell_portion

    def closed_form_mt_alteration(self, dstatesall_sm_, statesall_wsm, num_buys, num_sells, drdtB_, rsmB_, dt1B_, drdtS_, rsmS_, dt1s_, num_msm_, dstatesallstates_sm_, states_all_sm_, statesall2_wsm_, total_time):
        
        theta, k, eps, dsasm, sawsm, numb, nums, drdtB, rsmB, dt1B, drdtS, rsmS, dt1s, num_msm, dstatesallstates_sm, states_all_sm, statesall2_wsm, T = symbols('theta, k, eps, dsasm, sawsm, numb, nums, drdtB, rsmB, dt1B, drdtS, rsmS, dt1s, num_msm, dstatesallstates_sm, states_all_sm, statesall2_wsm, T')


        theta_opt = Eq((1 / (k * T)) * (dsasm + k * sawsm + eps * (nums - numb)) - theta, 0)
        eps_opt = Eq(drdtB - k * theta * numb + k * rsmB - eps * dt1B - drdtS + k * theta * nums - k * rsmS - eps * dt1s, 0)
        k_opt = Eq(theta * dsasm - T * (theta ** 2) * k + theta * k * sawsm - dstatesallstates_sm + theta * k * sawsm - k * statesall2_wsm + eps * rsmB - eps * theta * numb + eps * nums * theta - eps * rsmS, 0)


        substitutions = [(dsasm, dstatesall_sm_), (sawsm, statesall_wsm), (numb, num_buys), (nums, num_sells), (drdtB, drdtB_), (rsmB, rsmB_), (dt1B, dt1B_), (drdtS, drdtS_), (rsmS, rsmS_), (dt1s, dt1s_), (num_msm, num_msm_), (dstatesallstates_sm, dstatesallstates_sm_), (states_all_sm, states_all_sm_), (statesall2_wsm, statesall2_wsm_), (T, total_time)]
        theta_opt = theta_opt.subs(substitutions)
        eps_opt = eps_opt.subs(substitutions)
        k_opt = k_opt.subs(substitutions)


      

       # theta_opt = theta_opt.subs([(k, 1000), (eps, 20)])
       # print(solve(theta_opt, theta))
      #  print('attempting solve....')
        solved = solve([theta_opt, eps_opt, k_opt], theta, k, eps)
      #  print('stuck')
      #  print(solved)
        theta, k, eps = solved[0][0], solved[0][1], solved[0][2]

        var = self.omega_calc(k, theta, eps, self.dstatesbuy, self.cur_statesbuy, self.dstatessell, self.cur_statessell, self.dstatesnone, self.cur_statesnone, self.dtimesnone, self.dtimesbuy, self.dtimessell) / num_msm_
        return k, var, theta, eps


    def loss(self, x, delta_tbuy, delta_tsell, delta_tnone, dstatesbuy, curstatesbuy, dstatessell, curstatessell, dstatesnone, curstatesnone):
           
            k, var, theta, eps = x[0], x[1], x[2], x[3]

            k, var, theta, eps = float(k), float(var), float(theta), float(eps)
          #  print(k, var, theta, eps)
            buy_portion =  np.sum(-1 / (2 * var ** 2 *  delta_tbuy)  * (dstatesbuy - k * (theta - curstatesbuy) * delta_tbuy - eps) ** 2 -  0.5 * np.log(2 * math.pi * var ** 2 * delta_tbuy))
            sell_portion =  np.sum(-1 / (2 * var ** 2 *  delta_tsell)  * (dstatessell - k *(theta - curstatessell) * delta_tsell + eps) ** 2 -  0.5 * np.log(2 * math.pi * var ** 2 * delta_tsell)) 
            none_portion =  np.sum(-1 / (2 * var ** 2 *  delta_tnone)  * (dstatesnone - k *(theta - curstatesnone) * delta_tnone) ** 2 -  0.5 * np.log(2 * math.pi * var ** 2 * delta_tnone))

          #  var_der = -self.num_samples / (2 * var ** 2)  + (1 / (2 * var ** 4)) * self.omega_calc(k, theta, eps, self.dstatesbuy, self.cur_statesbuy, self.dstatessell, self.cur_statessell, self.dstatesnone, self.cur_statesnone, self.dtimesnone, self.dtimesbuy, self.dtimessell)
          #  theta_der = (k / var ** 2 ) * (np.sum(dstatesbuy - k * (theta - curstatesbuy) * delta_tbuy - eps) + np.sum(dstatessell - k *(theta - curstatessell) * delta_tsell + eps) + np.sum(dstatesnone - k *(theta - curstatesnone) * delta_tnone))
          #  eps_der = np.sum((dstatesbuy - k * (theta - curstatesbuy) * delta_tbuy - eps) / delta_tbuy) + np.sum((dstatessell - k * (theta - curstatessell) * delta_tsell - eps) / delta_tsell)
          #  k_der = np.sum((dstatesbuy - k * (theta - curstatesbuy) * delta_tbuy - eps) * (theta - curstatesbuy)) + np.sum((dstatessell - k * (theta - curstatessell) * delta_tsell - eps) * (theta - curstatessell)) + np.sum((dstatesnone - k *(theta - curstatesnone) * delta_tnone) * (theta - curstatesnone))
#
          #  print(k, var, theta, eps)
          #  print((k_der, var_der, theta_der, eps_der))
          #  print(-1 * (buy_portion + no_portion + sell_portion))
            return -1 * (buy_portion + none_portion + sell_portion) #, (k_der, var_der, theta_der, eps_der)


    def computational_maximization_mt(self):
        

        x0 = self.maxim()
      #  print(x0)
      #  print(x0)
     #   x0 = [100, 100, 100, 100]
        args_tuple = (self.dtimesbuy, self.dtimessell, self.dtimesnone, self.dstatesbuy, self.cur_statesbuy, self.dstatessell, self.cur_statessell, self.dstatesnone, self.cur_statesnone)
        
        bound = optim.Bounds((0.0000000001, 0.0000000001, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf))
        
        res = optim.minimize(self.loss, x0=x0, args=args_tuple, method="Newton-CG", jac=True, options={'maxiter' : 1000000, 'disp' : True})
      #  print(res.x)

        #print('starting')
        #res = optim.brute(loss, ranges=((18, 21), (0.01, 0.1), (-0.1, 0.1), (-0.1, 0.1)), args=args_tuple, Ns=2)
        #print('done')
        #print(res)




if __name__ == "__main__":
    dt = 0.00001
    simulation = Simulation(dt) 
    states, rec, bl, sl, mt = simulation.fit(dt)

    mle = MLE(rec, states, mt)
  #  print(mle.computational_maximization())
    print(mle.maxim())
  #  mle.computational_maximization_mt()
   # print(mle.computational_maximization())
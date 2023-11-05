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

import alpha_analysis as alpha
import hawkes_estimator as hawkes
import batched_k_estimator as fill_prob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
            order_rec = []
            buy_rec_times = []
            sell_rec_times = []
            k_plusses = []
            k_minusses = []
            alphas = []
            measurement_times = []
            volumes = []
            current_time = 0 

            
            b = 500
            theta = 20
            n = 400
            v = 70


            k_beta = 4000
            k_theta = 10000
            k_n = 1000
            k_v = 10


            zeta = 10
            sigma_alpha = 0.01
            epsilon = 0.05
            lt = 0
            alpha = lt

            volume_threshold = 0.8

            print(f'Theta Target: {k_theta} Beta Target: {k_beta} N Target: {k_n}, V Target: {k_v}')

            buy_rate = theta
            sell_rate = theta
            k_b = k_theta
            k_s = k_theta

            drift = lt

            while current_time <= 10:
                
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

                order_volume = np.random.standard_exponential(1)
                if rec_sell or rec_buy:
                    volumes.append(order_volume[0])
                else:
                    volumes.append(0)

                # can't receive two orders at once!
                if order_volume <= volume_threshold:
                    rec_sell_int = 0
                    rec_buy_int = 0





                if rec_buy and not rec_sell:
                    order_rec.append(1)
                    buy_rec_times.append(current_time)
                elif rec_sell and not rec_buy:
                    order_rec.append(-1)
                    sell_rec_times.append(current_time)
                elif not rec_sell and not rec_buy:
                    order_rec.append(0)
                elif rec_sell and rec_buy:
                    print("dude!")
                k_plusses.append(k_b)
                k_minusses.append(k_s)
                alphas.append(alpha)


                measurement_times.append(current_time)





                buy_rate += b * (theta - buy_rate) * dt + n * rec_buy_int + v * rec_sell_int
                sell_rate += b * (theta - sell_rate) * dt + n * rec_sell_int + v * rec_buy_int

                k_b += k_beta * (k_theta - k_b) * dt + k_n * rec_buy_int + k_v * rec_sell_int
                k_s += k_beta * (k_theta - k_s) * dt + k_v * rec_buy_int + k_n * rec_sell_int

                direction = np.random.randint(0, 2)
                if direction == 0:
                    alpha += zeta * (lt - alpha) * dt - sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                else:
                    alpha += zeta * (lt - alpha) * dt + sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                
                current_time += dt 


            return k_plusses, k_minusses, order_rec, measurement_times, buy_rec_times, sell_rec_times, alphas, volumes
        
        k_plusses, k_minusses, order_rec, measurement_times, buy_rec_times, sell_rec_times, alphas, volumes = generate_fitting_data(dt)
        print('done generating')
        plt.plot(alphas)
        plt.show()
        return np.array(k_plusses), np.array(k_minusses), np.array(order_rec), np.array(measurement_times), np.array(buy_rec_times), np.array(sell_rec_times), np.array(alphas), np.array(volumes)
    

class SupEstimator():

    def __init__(self, kp, km, oR, mt, br, sr, al, volumes):
        self.kp = kp
        self.km = km
        self.oR = oR
        self.mt = mt
        self.br = br
        self.sr = sr
        self.al = al
        self.volumes = volumes
        self.total_time = self.mt[-1] - self.mt[0]

        buy_indics = np.ones_like(self.br)
        sell_indics = -1 * np.ones_like(self.sr)

        trades = np.concatenate((self.br, self.sr))
        sorter = np.argsort(trades)
        trades = trades[sorter]
        self.trades = trades

        indics = np.concatenate((buy_indics, sell_indics))
        indics = indics[sorter]
        self.indics = indics
        influential = np.ones_like(indics)


        self.init_hawkes = hawkes.MLE(self.total_time, trades, indics, influential)
        self.init_alpha = alpha.MLE(self.oR, self.al, self.mt)
        self.init_k = fill_prob.MLE(self.kp, self.km, self.oR, self.mt)

        self.alpha_x0 = self.init_alpha.maxim()
        self.hawkes_x0 = self.init_hawkes.maxim_linear()
        self.k_x0 = self.init_k.maxim_linear()
        self.v_guessx0 = self.initial_threshold_guess()

        print(self.hawkes_x0, self.alpha_x0, self.k_x0, self.v_guessx0)

        self.threshold_final = self.v_guessx0[0]

        self.order_rec_f = np.copy(self.oR)
        self.order_rec_f[self.volumes < self.threshold_final] = 0 
        
        trade_volumes = self.volumes[self.volumes > 0]
        influential[trade_volumes < self.threshold_final] = 0 


        alpha_helper_f = alpha.MLE(self.order_rec_f, self.al, self.mt)
        hawkes_helper_f = hawkes.MLE(self.total_time, trades, indics, influential)
        k_helper_f = fill_prob.MLE(self.kp, self.km, self.order_rec_f, self.mt)

        self.alpha_final = alpha_helper_f.maxim()
        self.hawkes_final = hawkes_helper_f.maxim_linear()
        self.k_final = k_helper_f.maxim_linear()

        print(self.hawkes_final, self.alpha_final, self.k_final)



    def sup_estimation(self):
        print('going')
        def loss(x):

            cur_time = time.time()
            thresh, zeta_k, theta_k, n_k, v_k, zeta_alpha, theta_alpha, sigma_alpha, eps_alpha, beta_H, theta_H, n_H, v_H = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]
            
            measurement_time = self.mt
            order_rec = self.oR
            alphas = self.al
            dt = self.dt
            vols = self.volumes

            order_rec[vols <= thresh] = 0 
            trade_volumes = vols[vols > 0]
            influential = np.ones_like(self.indics)
            influential[trade_volumes < thresh] = 0 

            alpha_helper = alpha.MLE(order_rec, alphas, dt, measurement_time)
            k_helper = fill_prob.MLE(self.kp, self.km, order_rec, measurement_time)
            hawkes_helper = hawkes.MLE(self.total_time + self.dt, self.trades, self.indics, influential)

            alpha_loss = alpha_helper.loss([zeta_alpha, sigma_alpha, theta_alpha, eps_alpha], alpha_helper.dtimesbuy, alpha_helper.dtimessell, alpha_helper.dtimesnone, alpha_helper.dstatesbuy, alpha_helper.cur_statesbuy, alpha_helper.dstatessell, alpha_helper.cur_statessell, alpha_helper.dstatesnone, alpha_helper.cur_statesnone)
            k_loss = k_helper.loss([zeta_k, theta_k, n_k, v_k], k_helper.k_plusses, k_helper.k_minusses, k_helper.measurement_times, k_helper.order_rec)
            hawkes_loss = hawkes_helper.loss([beta_H, theta_H, n_H, v_H], hawkes_helper.trades, hawkes_helper.indicators, hawkes_helper.num_buys, hawkes_helper.num_sells, hawkes_helper.time_elapsed, hawkes_helper.influential)
            
            log_k_loss = np.log(k_loss)

            return hawkes_loss + alpha_loss + log_k_loss

        res = optim.minimize(fun=loss, x0=[self.v_guessx0[0], self.k_x0[0], self.k_x0[1], self.k_x0[2], self.k_x0[3], self.alpha_x0[0], self.alpha_x0[2], self.alpha_x0[1], self.alpha_x0[3], self.hawkes_x0[0], self.hawkes_x0[1], self.hawkes_x0[2], self.hawkes_x0[3]], bounds=((0, np.max(self.volumes)), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (-np.inf, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)), method="L-BFGS-B")
        print(res.x)

    def initial_threshold_guess(self):
        def threshold_loss(thresh):
            measurement_time = self.mt
            order_rec = np.copy(self.oR)
            alphas = self.al
            vols = self.volumes

            # if a trade occured, volume cannot be zero
            order_rec[vols <= thresh] = 0 
            trade_volumes = vols[vols > 0]
            influential = np.ones_like(self.indics)

           # print(order_rec.shape)
           # print(vols.shape)
           # print(trade_volumes.shape)
           # print(indicators.shape)


            influential[trade_volumes < thresh] = 0 

          #  print(np.count_nonzero(order_rec))
            alpha_helper = alpha.MLE(order_rec, alphas, measurement_time)
            k_helper = fill_prob.MLE(self.kp, self.km, order_rec, measurement_time)
            hawkes_helper = hawkes.MLE(self.total_time, self.trades, self.indics, influential)


            # the MLEs are already negated
            alpha_loss = alpha_helper.loss(self.alpha_x0, alpha_helper.dtimesbuy, alpha_helper.dtimessell, alpha_helper.dtimesnone, alpha_helper.dstatesbuy, alpha_helper.cur_statesbuy, alpha_helper.dstatessell, alpha_helper.cur_statessell, alpha_helper.dstatesnone, alpha_helper.cur_statesnone)
            k_loss = k_helper.loss(self.k_x0, k_helper.k_plusses, k_helper.k_minusses, k_helper.measurement_times, k_helper.order_rec)
            hawkes_loss = hawkes_helper.loss(self.hawkes_x0, hawkes_helper.trades, hawkes_helper.indicators, hawkes_helper.num_buys, hawkes_helper.num_sells, hawkes_helper.time_elapsed, hawkes_helper.influential)
            
            log_k_loss = np.log(k_loss)

            print(hawkes_loss + alpha_loss + log_k_loss)

            return hawkes_loss + alpha_loss + log_k_loss

        res = optim.minimize(fun=threshold_loss, x0=np.median(self.volumes[self.oR != 0]), method='Nelder-Mead', bounds=((0, np.inf),))
        return res.x 

    def get_params(self):
        return self.hawkes_final, self.alpha_final, self.k_final, self.v_guessx0

if __name__ == "__main__":
    dt = 0.00001
    simulation = Simulation(dt) 
    kp, km, oR, mt, br, sr, al, vols = simulation.fit(dt)
    supEst = SupEstimator(kp, km, oR, mt, br, sr, al, vols)
   # supEst.sup_estimation()
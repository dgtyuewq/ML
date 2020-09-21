import numpy as np
import matplotlib.pyplot as plt


x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# ydata = b + w * xdata

x = np.arange(-200,-100,1)#bias
y = np.arange(-5,5,0.1)#weight
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i]
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w*x_data[n])**2
        Z[j][i] = Z[j][i]/len(x_data)

#隨便找起始點
b = -120 # initial b
w = -4 # inital w
lr = 1 #learning rate 學習率
iteration = 100000

# Store initial values for plotting
b_history = [b] #所有b參數
w_history = [w] #所有w參數

lr_b = 0
lr_w = 0

# Iterations
for i in range(iteration):
    #我要找y=wx+b的解，y為預測值，y'為實際值
    #定義公式(實際值-預測值)**2就是(y'-(wx+b))**2
    #10個點就是L(w,b)=sig(10)[(y-(wx+b))**2]
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        #L(w,b)對b偏微分
        b_grad = b_grad -2.0*(y_data[n] - b - w*x_data[n])*1.0
        #L(w,b)對w偏微分
        w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    #Adagrad修改learning rate
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    
    # Update parameters.
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_grad
    
    # Store parameters or plotting
    b_history.append(b)
    w_history.append(w)

# plor the figure
plt.contourf(x,y,Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67], 'x', ms=12, markeredgewidth=3, color = 'orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
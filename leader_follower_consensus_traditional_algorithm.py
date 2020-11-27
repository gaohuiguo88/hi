import numpy as np
adj=[[0,0,0,0,0,0,0],[1,0,0,1,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,1],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0]]

A = [[0,1,0],[0,0,1],[0,0,0]]
B = [[0],[0],[1]]
K = [-0.6285,-1.3525,-2.1113]
P = [[3.0861,-0.6245,-0.5186],[-0.6245,1.1602,-0.5573],[-0.5186,-0.5573,0.9850]]
Gamma = [[0.3950,0.8500,1.3269],[0.85,1.8292,2.8554],[1.3269,2.8554,4.4574]]
x = np.random.random(7)
u = np.zeros(7)
# calculate the control input
for i in range(1,len(u)):
    s = 0
    for j in range(len(u)):
        s += adj[i][j] * (x[i] - x[j])
    u[i] = c * np.matmul(K,s)

# update
for i in range(len(x)):
    x[i] = x[i] + np.matmul(A,x[i]) + np.matmul(B,u[i])
    

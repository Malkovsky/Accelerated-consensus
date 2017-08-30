from cvxpy import *
from matplotlib.pyplot import * 
import numpy as np
import math
import sys

reload(sys)  
sys.setdefaultencoding('utf8')

eps = 0.001

if len(sys.argv) > 1:
	f = open(sys.argv[1], "r")
else:
	f = open(".in", "r")
_n, _m = f.readline().split()
n = int(_n)
m = int(_m)
print(n, m)
mu = 2 * (1 - math.cos(math.pi / n))

steps = 3 * n


inc = np.zeros((n, m))
mtrx = list()

def sqr(z):
	return z * z

def cons_norm(z):
	e = 0
	sm = 0
	for t in z:
		sm += t
	sm /= n
	#print('??',sm, z)
	for t in z:
		e += (t - sm) * (t - sm)
	e = math.sqrt(e)
	return e

def simple(P, z, exact = False):
	alpha = 0.5
	if exact:
		alpha = 2 / (sum(t for t in np.linalg.eigvalsh(P)[1:n:n-2]))
	else:
		D = np.linalg.inv(np.diag(np.diag(P)))
		P = np.dot(D, P)

	err1 = list()
	for i in range(steps):
		err1.append(cons_norm(z)) 
		z = z - np.dot(P, z) * alpha
	return err1


def heavy_ball(P, z, mu, Mu):
	err2 = list()
	z1 = z.copy()
	alpha = 4 / sqr(math.sqrt(mu)+math.sqrt(Mu))
	beta = sqr((math.sqrt(Mu)-math.sqrt(mu)) / (math.sqrt(Mu)+math.sqrt(mu)))

	for i in range(steps):
		err2.append(cons_norm(z)) 
		z2 = z.copy()
		z = z - np.dot(alpha, np.dot(P, z)) + np.dot(beta, z - z1)
		z1 = z2

	return err2

def nesterov(P, z, mu, Mu):
	y = z.copy()
	z1 = z.copy()
	alpha2 = 1 / Mu
	beta2  = (math.sqrt(Mu)-math.sqrt(mu)) / (math.sqrt(Mu)+math.sqrt(mu))
	err3 = list()
	for i in range(steps):
		err3.append(cons_norm(z))
		z1 = y - np.dot(alpha2, np.dot(P, y))
		y = z1 + np.dot(beta2, z1 - z);
		z = z1.copy()
	return err3
		

for i in range(m): 
	_q, _w = f.readline().split()
	q, w = int(_q), int(_w)
	inc[q - 1][i] = -1
	inc[w - 1][i] = 1
	#print(np.zeros((n, n)))

J = np.ones((n, n)) / n;

s = Variable(1)
x = Variable(m)
#L = Variable(n, n) 

I = np.diag(np.ones(n))


cs = [x >= 0, s * I << inc * diag(x) * inc.T + J, inc * diag(x) * inc.T + J << I]

prob = Problem(Maximize (s), cs)
print prob.solve()


#non conditioned
#simple
Ls = np.dot(np.dot(inc, np.diag(np.ones(m))), inc.transpose())
D = np.diag(np.diag(Ls))
Mu = 0
for t in D:
	for tt in t:
		if Mu < tt:
			Mu = tt
Mu = 2 * Mu
print('Predicted condition number:', Mu / mu)
print('Predicted eigenbounds', mu, Mu)
print('Real eigenbounds', np.linalg.eigvalsh(np.array(Ls))[1:n:n-2])
#mu, Mu = np.linalg.eigvalsh(np.array(Ls))[1:n:n-2]

D = np.linalg.inv(D)
z = np.ones(n)
z[0] = n
err1 = simple(Ls, z)

#heavy ball
z = np.ones(n)
z[0] = n
err2 = heavy_ball(Ls, z, mu, Mu)

#Nesterov
z = np.ones(n)
z[0] = n
err3 = nesterov(Ls, z, mu, Mu)


#for i in range(steps)[95:steps:1]:
#	print ('#{0:2d}: {1:.8f} {2:.8f} {3:.8f}'.format(i, err1[i], err2[i], err3[i]))

ptwdth=1.5
fig, ax = subplots()
ax.plot(err1, label='Base consensus', ls=':', color='b', lw = ptwdth)
ax.plot(err2, label='Heavy ball', ls='-', color='b', lw = ptwdth)
ax.plot(err3, label='Nesterov', ls='--', color='b', lw = ptwdth)

#conditioned + exact eigenbounds
#print('L', L.value)
#print('x', np.array(x.T.value)[0])
#print(inc, np.diag(np.array(x.T.value)[0]), inc.transpose()) 
A = np.dot(np.dot(inc, np.diag(np.array(x.T.value)[0])), inc.transpose())
#print('A', A)
#mu, Mu = np.linalg.eigvalsh(np.array(L.value))[1:n:n-2]
mu = s.value
Mu = 1
print('s', s.value)
print('Eigenbounds:', mu, Mu)
print('Eigenbounds:', np.linalg.eigvalsh(A)[1:n:n-2]) 
print('Condition number:', Mu / mu)
#mu = mu / (1 + eps)
#Mu = Mu * (1 + eps)

#D = np.linalg.inv(np.diag(np.diag(L.value)))
z = np.ones(n)
z[0] = n
err1 = simple(np.array(A), z, exact = True)

#heavy ball
z = np.ones(n)
z[0] = n
err2 = heavy_ball(np.array(A), z, mu, Mu)

#Nesterov
z = np.ones(n)
z[0] = n
err3 = nesterov(np.array(A), z, mu, Mu)

#for i in range(steps)[95:steps:1]:
#	print ('#{0:2d}: {1:.8f} {2:.8f} {3:.8f}'.format(i, err1[i], err2[i], err3[i]))

ax.plot(err1, label='Base consensus + cond', ls=':', color='r', lw = ptwdth)
ax.plot(err2, label='Heavy ball + cond', ls='-', color='r', lw = ptwdth)
ax.plot(err3, label='Nesterov + cond', ls='--', color='r', lw = ptwdth)

legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')

show()





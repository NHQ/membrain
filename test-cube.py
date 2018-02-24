import numpy as np
import tensorflow as tf

#cubic_model = tf.pow((1 - t), [3]) * cp0 + tres * tf.pow(1 - t, [2]) * t * cp1 + tres * (1 - t) * tf.pow(t, [2]) * cp2 + tf.pow(t, [3]) * cp3

samplerate = 16 
loops = 50000

time = np.arange(0, samplerate, dtype='float32') / samplerate

t = tf.constant(np.asarray(time), 'float32', shape=[samplerate,1])

nt = 1 - t

cpt0 = tf.pow(nt, 3) 
cpt1 = tf.pow(nt, 2) * 3 * t
cpt2 = nt * 3 * tf.pow(t, 2)
cpt3 = tf.pow(t, 3)

time_stack = tf.concat([cpt0, cpt1, cpt2, cpt3], 1)

x = [ 0.8791666666666667,
  0.6663523356119792,
  0.5186442057291666,
  0.42683207194010414,
  0.3817057291666666,
  0.37405497233072915,
  0.39466959635416665,
  0.4343393961588541,
  0.4838541666666666,
  0.5340037027994791,
  0.5755777994791667,
  0.5993662516276042,
  0.5961588541666666,
  0.5567454020182291,
  0.4719156901041667,
  0.3324595133463542
  ] 

y = [ 0.15416666666666667,
  0.35943705240885415,
  0.5295328776041667,
  0.6661936442057294,
  0.7711588541666669,
  0.8461680094401042,
  0.8929606119791668,
  0.9132761637369793,
  0.9088541666666667,
  0.8814341227213541,
  0.8327555338541667,
  0.7645579020182292,
  0.6785807291666666,
  0.5765635172526042,
  0.4602457682291667,
  0.3313669840494792
  ]


solve_X = tf.constant(x, 'float32', shape=[1,samplerate])
solve_Y = tf.constant(y, 'float32', shape=[1,samplerate])

sess = tf.Session()
sess.run([time_stack, solve_X, solve_Y])

X = tf.get_variable('cpx', [4,], 'float32', tf.random_normal_initializer())
Y = tf.get_variable('cpy', [4,], 'float32', tf.random_normal_initializer())
cubic_X = tf.reduce_sum(tf.matmul(time_stack, tf.diag(X)) , 1)
cubic_Y = tf.reduce_sum(tf.matmul(time_stack, tf.diag(Y)) , 1)

loss_X = solve_X - cubic_X
loss_Y = solve_Y - cubic_Y

loss = tf.sqrt(tf.reduce_mean(tf.pow(tf.concat([loss_X, loss_Y], 1), 2), 1))

train = tf.train.AdamOptimizer(1e-4).minimize(loss)

def cond(i, x, y):
  return i < loops 

def algo(i, x, y):
  # do the loss loop in here
  train.minimize(loss)

  return i + 1

#loop = tf.while_loop(cond, algo, (0, X, Y))
sess.run(tf.global_variables_initializer())
_, __, l = sess.run([X, Y, loss])
l = l[0] #sess.run(loss)[0]
print l
cycles = 0
#i, x, y = sess.run(loop)
while(l > .000123333):
  l, _, _X, _Y = sess.run([loss, train, X, Y])
  l = l[0]
  cycles+=1


print l, cycles, _X, _Y
#for _ in range(loops):
#  sess.run(train)
#print x, y
#a,b,c = sess.run([X, Y, loss])
#print a
#print b
#print c
#print sess.run(control_points_X)

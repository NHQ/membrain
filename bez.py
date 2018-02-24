import numpy as np
import tensorflow as tf
from scipy import special as sp

#cubic_model = tf.pow((1 - t), [3]) * cp0 + tres * tf.pow(1 - t, [2]) * t * cp1 + tres * (1 - t) * tf.pow(t, [2]) * cp2 + tf.pow(t, [3]) * cp3

def izOne(x, y):
  if x == 1: 
    return 1
  if y == 1: 
    return 1
  else: 
    return x * y

def bez(solve, shape=1, size=16, scale=4, dims=2):

  Time = np.arange(0, size, dtype='float32') / size

  t = tf.constant(np.asarray(Time), 'float32', shape=[shape, size, 1, dims])

  for i in range(shape - 1):
    t = t.concat([t, t], 0)

  nt = 1 - t


  # time stack == first order of time stack, below
  # you did this to make the for loop + concat smooth

  time_stack = tf.pow(nt, scale - 1)
  
  for i in range(scale):
    if i == 0 or i == scale - 1:
      continue
    else:
      bi = sp.binom(scale - 1, i)
      computation = tf.pow(nt, scale - i) * bi * tf.pow(t, i)
      time_stack = tf.concat([time_stack, computation], 1) 

  # concat last order of time stack, below
  # should be correct

  time_stack = tf.concat([time_stack, tf.pow(t, scale - 1)], 1)


  sess = tf.Session()
  sess.run([time_stack, solve_X, solve_Y])

  X = tf.get_variable('cpx', [4,], 'float32', tf.random_normal_initializer())
  Y = tf.get_variable('cpy', [4,], 'float32', tf.random_normal_initializer())
  cubic_X = tf.reduce_sum(tf.matmul(time_stack, tf.diag(X)) - 1, 1)
  cubic_Y = tf.reduce_sum(tf.matmul(time_stack, tf.diag(Y)) - 1, 1)

  loss_X = solve_X - cubic_X
  loss_Y = solve_Y - cubic_Y

  loss = tf.sqrt(tf.reduce_mean(tf.pow(tf.concat([loss_X, loss_Y], 1), 2), 1))

  train = tf.train.AdamOptimizer(1e-4).minimize(loss)

  def cond(i, x, y):
    return i < loops 

  def algo(i, x, y):
    # do the loss loop in here
    train.minimize(loss)

    return (i + 1, x, y)

  #loop = tf.while_loop(cond, algo, (0, X, Y))

  sess.run(tf.global_variables_initializer())
  print sess.run([X, Y, loss])
  #i, x, y = sess.run(loop)
  for _ in range(loops):
    sess.run(train)
  #print x, y
  a,b,c = sess.run([X, Y, loss])
  print a
  print b
  print c
  #print sess.run(control_points_X)
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
  0.3324595133463542]

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
  0.3313669840494792]

solve_X = tf.constant(x, 'float32', shape=[shape, 1,size])
solve_Y = tf.constant(y, 'float32', shape=[shape, 1,size])

bez([x,y])

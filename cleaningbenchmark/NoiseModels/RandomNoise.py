"""
This module defines a collection of random
noise modules--that is they do not use the
features.
"""
import numpy
import NoiseModel
import pandas


def bound_number(value, low, high):
  return max(low, min(high, value))

vbound = numpy.vectorize(bound_number)


"""
This model implements Gaussian Noise
"""
class GaussianNoiseModel(NoiseModel.NoiseModel):
  
  """
  Mu and Sigma are Params
  """
  def __init__(self, 
               shape, 
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               mu=0,
               sigma=1,
               scale=numpy.array([]),
               int_cast=numpy.array([])):

    super(GaussianNoiseModel, self).__init__(shape, 
                                             probability, 
                                             feature_importance,
                                             one_cell_flag)
    self.mu = mu
    self.sigma = sigma
    self.scale = scale # numpy array

    # cast the noise quantity to int, if True
    if not int_cast.size: 
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:
      if not self.scale.size:
        scale = 1.0
      Z = numpy.random.randn(Ns,ps)*self.sigma*scale + self.mu
      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in xrange(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
        else:
          scale = self.scale[a]

        if self.int_cast[a]:
          noiz = int(numpy.ceil(numpy.random.randn()*self.sigma*scale + self.mu))
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = numpy.random.randn()*self.sigma*scale + self.mu
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):
 
    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(numpy.random.randn()*self.sigma*scale + self.mu))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
      else:
        noiz = numpy.random.randn()*self.sigma*scale + self.mu
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)



"""
This model implements Laplace Noise
"""
class LaplaceNoiseModel(NoiseModel.NoiseModel):
  
  """
  Mu and Sigma are Params
  """
  def __init__(self, 
               shape, 
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               mu=0,
               b=1,
               scale=numpy.array([]),
               int_cast=numpy.array([])):

    super(LaplaceNoiseModel, self).__init__(shape, 
                                            probability, 
                                            feature_importance,
                                            one_cell_flag)
    self.mu = mu
    self.b = b
    self.scale = scale # numpy array

    # cast the noise quantity to int, if True
    if not int_cast.size: 
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast
  
  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:
      if not self.scale.size:
        scale = 1.0
      Z = numpy.random.laplace(self.mu, self.b, (Ns,ps))*scale
      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in xrange(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
        else:
          scale = self.scale[a]

        if self.int_cast[a]:
          noiz = int(numpy.ceil(numpy.random.laplace(self.mu, self.b)*scale))
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = numpy.random.laplace(self.mu, self.b)*scale
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):
 
    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(numpy.random.laplace(self.mu, self.b)*scale))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
      else:
        noiz = numpy.random.laplace(self.mu, self.b)*scale
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)


"""
Zipfian Noise, simulates high-magnitude outliers
"""
class ZipfNoiseModel(NoiseModel.NoiseModel):
  
  """
  z is the Zipfian Scale Parameter
  """
  def __init__(self, 
               shape, 
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               z=3,
               scale=numpy.array([]),
               int_cast=numpy.array([]),
               active_neg=False):

    super(ZipfNoiseModel, self).__init__(shape, 
                                         probability, 
                                         feature_importance,
                                         one_cell_flag)

    self.z = z
    self.scale = scale # numpy array

    # cast the noise quantity to int, if True
    if not int_cast.size: 
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast

    # flag to generate negative values also
    # zipf_val*numpy.random.choice([-1,1])
    self.active_neg = active_neg

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:
      if not self.scale.size:
        scale = 1.0
      Z = numpy.random.zipf(self.z, (Ns,ps))*scale
      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in xrange(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
        else:
          scale = self.scale[a]

        if self.active_neg:
          sign_val = numpy.random.choice([-1,1])
        else:
          sign_val = 1

        if self.int_cast[a]:
          noiz = int(numpy.ceil(numpy.random.zipf(self.z)*scale))
          Y[i,a] = bound_number(Y[i,a] + noiz*sign_val, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = numpy.random.zipf(self.z)*scale
          Y[i,a] = bound_number(Y[i,a] + noiz*sign_val, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):
 
    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]

      if self.active_neg:
        sign_val = numpy.random.choice([-1,1])
      else:
        sign_val = 1

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(numpy.random.zipf(self.z)*scale))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz*sign_val, numpy.iinfo(int).min, numpy.iinfo(int).max)

      else:
        noiz = numpy.random.zipf(self.z)*scale
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz*sign_val, numpy.finfo(float).min, numpy.finfo(float).max)
        

"""
Simulates Random Errors for Categorical Features
Inserts both Typos and Category Changes
"""
class CategoricalNoiseModel(NoiseModel.NoiseModel):

  """
  The order in 'cats_name_lists' should be the same as in
  'cats_probs_list', such that for each column/feature their
  category names and probabilities match
  """
  
  def __init__(self,
               shape,
               cats_name_lists,
               probability=0,
               feature_importance=[],
               cats_probs_list=[], 
               typo_prob=0.01,
               alpha_prob=1.0):

    super(CategoricalNoiseModel, self).__init__(shape,
                                                probability, 
                                                feature_importance,
                                                True)

    self.cats_name_lists = cats_name_lists
    self.cats_probs_list = cats_probs_list
    self.typo_prob = typo_prob
    self.alpha_prob = alpha_prob

    if not self.cats_probs_list:
      # if cats_probs_list not provided assume that each feature/column has 
      #   uniform distribution on its categories
      self.cats_probs_list = [numpy.ones(len(self.cats_name_lists[i])) / float(len(self.cats_name_lists[i])) 
                              for i in xrange(len(self.cats_name_lists))]

    self.cats_probs_list = [self.cats_probs_list[i]**alpha_prob / numpy.sum(self.cats_probs_list[i]**alpha_prob)
                            for i in xrange(len(self.cats_name_lists))]

    # renormalize categorical probabilities for each feature
    #   as to include typo probability
    self.cats_probs_list = [numpy.append(cats_prob*(1-typo_prob), typo_prob) 
                            for cats_prob in self.cats_probs_list]

  def corrupt(self, X):
    """
    X must be ndarray (numpy) with dtype object

    NOTE: To obtain Uniform distribution across all choices (Categories + Typos) do: 
          -> cats_probs_list=[]
          -> typo_prob=1/(N_categories + 1)
    """

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]
    Y = numpy.copy(X)

    for i in range(0, Ns):

      a = numpy.random.choice(ps)

      tmp_cat_name_list = self.cats_name_lists[a] + [NoiseModel.generate_typo(str(Y[i,a]))]
      tmp_cat_prob_list = self.cats_probs_list[a]

      idx_rmv = -1
      for idx, elem in enumerate(tmp_cat_name_list):
        if elem == Y[i,a]:
          idx_rmv = idx
          break

      if idx_rmv >= 0:
        tmp_cat_name_list.pop(idx_rmv)
        tmp_cat_prob_list = numpy.delete(self.cats_probs_list[a], idx_rmv)
        tmp_cat_prob_list = tmp_cat_prob_list / tmp_cat_prob_list.sum()

      Y[i,a] = numpy.random.choice(tmp_cat_name_list, 1, False, tmp_cat_prob_list)[0]

    return Y

  def corrupt_elem(self, Y, idxs, idx_cat_map):
 
    for idx_0, idx_1 in idxs:

      idx_cat = idx_cat_map[idx_1]

      tmp_cat_name_list = self.cats_name_lists[idx_cat] + [NoiseModel.generate_typo(str(Y[idx_0,idx_1]))]
      tmp_cat_prob_list = self.cats_probs_list[idx_cat]

      idx_rmv = -1
      for idx, elem in enumerate(tmp_cat_name_list):
        if elem == Y[idx_0,idx_1]:
          idx_rmv = idx
          break

      if idx_rmv >= 0:
        tmp_cat_name_list.pop(idx_rmv)
        tmp_cat_prob_list = numpy.delete(self.cats_probs_list[idx_cat], idx_rmv)
        tmp_cat_prob_list = tmp_cat_prob_list / tmp_cat_prob_list.sum()

      Y[idx_0,idx_1] = numpy.random.choice(tmp_cat_name_list, 1, False, tmp_cat_prob_list)[0]


"""
Simulates Mixed (Numerical and Categorical) Noise Injection
"""
class MixedNoiseTupleWiseModel(NoiseModel.NoiseModel):

  def __init__(self, 
               shape, 
               cat_array_bool,
               idx_map_cat,
               idx_map_num,
               model_categorical,
               model_numerical,
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               p_row=0.10):

    super(MixedNoiseTupleWiseModel, self).__init__(shape, 
                                                   probability, 
                                                   feature_importance,
                                                   one_cell_flag)
    
    self.model_cat = model_categorical
    self.model_num = model_numerical

    self.idx_map_cat = idx_map_cat # TODO: check if this is good input, should the overall processing be placed inside ?!!
    self.idx_map_num = idx_map_num # TODO: check if this is good input, should the overall processing be placed inside ?!! 

    self.cat_array_bool = cat_array_bool
    self.p_row = p_row

  def corrupt(self, X):
    """
    X must be ndarray (numpy)
    """

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]
    Y = numpy.copy(X)

    # generate idxs to be used in noising (from the selected dataset)
    idx_mat = numpy.random.uniform(0.0, 1.0, X.shape) <= self.p_row 

    # get categorical indexes for dirty cells
    idxs_cat = numpy.where(idx_mat & self.cat_array_bool)

    # get numerical indexes for dirty cells
    idxs_num = numpy.where(idx_mat & numpy.logical_not(self.cat_array_bool))

    # set dirty values into Y
    self.model_cat.corrupt_elem(Y, zip(idxs_cat[0], idxs_cat[1]), self.idx_map_cat)
    self.model_num.corrupt_elem(Y, zip(idxs_num[0], idxs_num[1]), self.idx_map_num)

    return Y


"""
Simulates Random Missing Data With a Placeholder Value
Picks an attr at random and sets the value to be missing
"""
class MissingNoiseModel(NoiseModel.NoiseModel):
  
  """
  ph is the Placeholder value that missing attrs are set to.
  """
  def __init__(self, 
               shape, 
               probability=0,
               feature_importance=[],
               ph=-1):

    super(MissingNoiseModel, self).__init__(shape, 
                                   probability, 
                                   feature_importance,
                                   True)
    self.ph = ph
  

  def corrupt(self, X):
    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]
    Y = numpy.copy(X)
    for i in range(0, Ns):
      a = numpy.random.choice(ps,1)
      Y[i,a] = self.ph
    return Y

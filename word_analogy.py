# Corbett Moore
# LING 570
# HW 10: word_analogy
# Due December 7

import sys
import math
import numpy as np
from datetime import datetime

vector_file = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
normalize_flag = int(sys.argv[4])
similarity = int(sys.argv[5])

symbol2index = {}
lines = []
width = 0
index2symbol = []


# Read in all of the initial vectors into an array, first converting each word into an index for that array.

def read_vectors():
    global width
    vectorgoodies = open(vector_file, 'r')
    for line in vectorgoodies:  # We are going to first read all lines into memory
        lines.append(line)
        if width == 0:
            words = line.split()
            width = len(words)


# Here we read in the vectors and establish a correctly-sized array in Numpy

print(datetime.time(datetime.now()))
read_vectors()
vectors = np.zeros(shape=(len(lines), width - 1))
mags = np.zeros(shape=len(lines))

# We process the lines as vectors and put them into the Numpy array

def calc_vectors():
    for line in lines:
        words = line.split()
        idx = symbol_to_index(words[0])
        nums = []
        mag = 0
        mag2 = 0
        for i in range(1, len(words)):
            val = float(words[i])
            mag += val * val
            nums.append(val)
        mag = math.sqrt(mag)  # We assign the magnitude to an array for reference
        if normalize_flag == 0:
            for i in range(len(nums)):
                nums[i] = nums[i]/mag
                mag2 += nums[i] * nums[i]
            mag = math.sqrt(mag2)
        mags[idx] = mag
        if len(nums) != width - 1:  # We have run across an incompatible line length
            sys.exit("Incompatible line length on word {0} [{1}]".format(idx, words[0]))
        vectors[idx] = nums



# Converts a word to an index that can be used to reference the array, and a reverse-lookup of index to word.

def symbol_to_index(text):
    if text not in symbol2index:
        symbol2index[text] = len(symbol2index)
        index2symbol.append(text)
    return symbol2index[text]


# Read in all files in the input directory, sends each one to the read_questions function

def iterate_directory():
    from os import listdir
    from os.path import isfile, join
    thisdir = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    all_files = len(thisdir)
    total_right = 0
    total_lines = 0
    for i in range(all_files):
        vals = read_questions(thisdir[i])
        sys.stdout.write("{0}\nACCURACY TOP1: {1}% ({2}/{3})\n".format(thisdir[i], vals[0] / vals[1], vals[0], vals[1]))
        total_right += vals[0]
        total_lines += vals[1]
    sys.stdout.write("Total accuracy: {0}% ({1}/{2})\n".format(total_right/total_lines, total_right, total_lines))



#  Reads in the questions in the given file

def read_questions(file):
    inputgoodies = open("{0}/{1}".format(input_dir,file), 'r')
    outputgoodies = open("{0}/{1}".format(output_dir,file), 'w')
    num_right = 0
    total_num = 0
    for line in inputgoodies:
        total_num += 1
        words = line.split()
        vector_a = zero_vector()
        vector_b = zero_vector()
        vector_c = zero_vector()
        if words[0] in symbol2index:  # Here we only want to read in the input questions, not make new indices
            idx = symbol_to_index(words[0])
            vector_a = vectors[idx]
        if words[1] in symbol2index:
            idx = symbol_to_index(words[1])
            vector_b = vectors[idx]
        if words[2] in symbol2index:
            idx = symbol_to_index(words[2])
            vector_c = vectors[idx]
        d_idx = find_vector(vector_a, vector_b, vector_c)  # Gets the index number of the best vector
        d_word = index2symbol[d_idx]
        outputgoodies.write("{0} {1} {2} {3}\n".format(words[0], words[1], words[2], d_word))
#        print("{0} : {1} :: {2} : {3}".format(words[0], words[1], words[2], d_word))
        if d_word == words[3]:
            num_right += 1
    return num_right, total_num


# Turns a line of input into an appropriate vector. When flag1 == 0, we normalize (sqrt of the sum of all
# squares of each dimension in the vector). When flag1 != 0, we use the vector as given.

def make_vector(word):
    if word in symbol2index:
        idx = symbol_to_index(word)
        if normalize_flag == 0:  # Per flag1, we are normalizing
            return normalize_vec(idx)
        else:
            return vectors[idx]
    elif word not in symbol2index:  # If word not found, return a 0-vector of the appropriate length
        return zero_vector()


# Returns a zero vector the same length as vectors[0]

def zero_vector():
    new_vector = []
    for i in range(len(vectors[0])):
        new_vector.append(0)
    return new_vector


# A routine to normalize the vector given

def normalize_vec(idx):
    z = mags[idx]
    if z != 0:
        new_vector = []
        for i in range(len(vectors[idx])):
            item = vectors[idx]
            new_vector.append(item[i]/z)
        return new_vector
    else:  # If the magnitude == 0, we cannot normalize so we return a zero vector instead
        return zero_vector()


# Directs to the appropriate finding-vector routine

def find_vector(vector_a, vector_b, vector_c):
    proposed_d = np.add(vector_b, vector_c)
    proposed_d = np.subtract(proposed_d, vector_a)
    if similarity == 0:  # Per flag2, we are finding Euclidean distance
        return find_euclidean(proposed_d)
    else:  # Per flag2, we are finding cosine similarity
        return find_cosine(proposed_d)


# Find the closest vector by Euclidean distance.

def find_euclidean(proposed_d):
    deltas = vectors - proposed_d
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)



#def find_euclidean(proposed_d):
#    best = float("+inf")  # Starting at +INF, the closest distance
#    bestidx = -1  # The index of the best vector
#    for j in range(len(vectors)):
#        tot = 0
#        this_vector = vectors[j]
#        for k in range(len(this_vector)):
#            diff = proposed_d[k] - this_vector[k]
#            tot += diff * diff
#        squareroot = math.sqrt(tot)
#        print("Best: {0} vs {1}".format(best, squareroot))
#        if squareroot < best:
#            best = squareroot
#            bestidx = j
#    return bestidx


# Find the closest vector using cosine similarity.

def find_cosine(proposed_d):
    dots = np.dot(vectors, proposed_d)
    mag_d = vector_mag(proposed_d)
    denoms = np.linalg.norm(vectors) * mag_d
    deltas = dots/denoms
    best = -1
    bestidx = -1
    for i in range(len(deltas)):
        if deltas[i] > best:
            best = deltas[i]
            bestidx = i
    return bestidx

#def find_cosine(proposed_d):
#    best = -1  # Starting at -1, the worst possible value
#    bestidx = -1
#    mag_d = vector_mag(proposed_d)
#    for i in range(len(vectors)):
#        tot = 0
#        mag_this = mags[i]
#        this_vector = vectors[i]
#        for j in range(len(vectors[i])):
#            tot += this_vector[j] * proposed_d[j]
#        tot = np.dot(this_vector, proposed_d)
#        cossim = tot / (mag_d * mag_this)
#        print("Best: {0} vs {1}".format(best, cossim))
#        if cossim > best:
#            best = cossim
#            bestidx = i
#    return bestidx


# Finds a vector's magnitude. I could have taken this from numpy, but I didn't.

def vector_mag(nums):
    tot = 0
    for i in range(len(nums)):
        tot += nums[i] * nums[i]
    z = math.sqrt(tot)
    return z


# Calculate all the magnitudes

def calc_magnitudes():
    for i in range(len(vectors)):
        mags.append(0)
        mags[i] = vector_mag(vectors[i])


calc_vectors()
iterate_directory()
print(datetime.time(datetime.now()))
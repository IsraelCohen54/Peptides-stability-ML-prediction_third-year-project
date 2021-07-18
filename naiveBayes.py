import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import Subset
import math
import torch
from mainMLP import makeInputData

AMINO_ACIDS = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')


def myEncodeDense(seqs, labels):
    def myEncodeDenseOne(seq):
        return [(seq == AA).sum() for AA in AMINO_ACIDS]

    seqs_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in seqs])
    labels_encoded = torch.FloatTensor(labels)
    return seqs_encoded, labels_encoded


def shuffle_data(seqs, labels):
    return shuffle(seqs, labels, random_state=0)


def fill_arrays_data(sequence, the_label):
    #0-19, for stab, 20 - 39 = not stab
    one_AA = np.ones(40)
    two_AA = np.ones(40)
    three_AA = np.ones(40)
    four_AA = np.ones(40)
    five_AA = np.ones(40)
    six_AA = np.ones(40)
    seven_AA = np.ones(40)
    eight_AA = np.ones(40)
    nine_AA = np.ones(40)
    ten_AA = np.ones(40)
    eleven_AA = np.ones(40)
    twelve_AA = np.ones(40)
    thirteen_AA = np.ones(40)
    fourteen_AA = np.ones(40)
    fifteen_AA = np.ones(40)

    for index, one_seq in enumerate(sequence):
        if the_label[index] == 1:
            for sub_seq_ind, letter in enumerate(one_seq):
                letter = int(letter)
                if letter == 0:
                    continue
                if letter == 1:
                    one_AA[sub_seq_ind] += 1
                    continue
                if letter == 2:
                    two_AA[sub_seq_ind] += 1
                    continue
                if letter == 3:
                    three_AA[sub_seq_ind] += 1
                    continue
                if letter == 4:
                    four_AA[sub_seq_ind] += 1
                    continue
                if letter == 5:
                    five_AA[sub_seq_ind] += 1
                    continue
                if letter == 6:
                    six_AA[sub_seq_ind] += 1
                    continue
                if letter == 7:
                    seven_AA[sub_seq_ind] += 1
                    continue
                if letter == 8:
                    eight_AA[sub_seq_ind] += 1
                    continue
                if letter == 9:
                    nine_AA[sub_seq_ind] += 1
                    continue
                if letter == 10:
                    ten_AA[sub_seq_ind] += 1
                    continue
                if letter == 11:
                    eleven_AA[sub_seq_ind] += 1
                    continue
                if letter == 12:
                    twelve_AA[sub_seq_ind] += 1
                    continue
                if letter == 13:
                    thirteen_AA[sub_seq_ind] += 1
                    continue
                if letter == 14:
                    fourteen_AA[sub_seq_ind] += 1
                    continue
                if letter == 15:
                    fifteen_AA[sub_seq_ind] += 1
                    continue
        else:
            for sub_seq_ind, letter in enumerate(one_seq):
                letter = int(letter)
                if letter == 0:
                    continue
                if letter == 1:
                    one_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 2:
                    two_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 3:
                    three_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 4:
                    four_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 5:
                    five_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 6:
                    six_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 7:
                    seven_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 8:
                    eight_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 9:
                    nine_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 10:
                    ten_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 11:
                    eleven_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 12:
                    twelve_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 13:
                    thirteen_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 14:
                    fourteen_AA[sub_seq_ind+20] += 1
                    continue
                if letter == 15:
                    fifteen_AA[sub_seq_ind+20] += 1
                    continue
    return (
    one_AA, two_AA, three_AA, four_AA, five_AA, six_AA, seven_AA, eight_AA, nine_AA, ten_AA, eleven_AA, twelve_AA,
    thirteen_AA, fourteen_AA, fifteen_AA)

def check_prior(one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,AA_index,array_num_chooser):
    stab_prior_prob = 0
    not_stab_prior_prob = 0
    stab_data = 0
    non_stab_data = 0
    array_num_chooser = int(array_num_chooser)
    if array_num_chooser == 1:
        stab_data = one[AA_index]
        non_stab_data = one[AA_index+20]
    elif array_num_chooser == 2:
        stab_data = two[AA_index]
        non_stab_data = two[AA_index+20]
    elif array_num_chooser == 3:
        stab_data = three[AA_index]
        non_stab_data = three[AA_index+20]
    elif array_num_chooser == 4:
        stab_data = four[AA_index]
        non_stab_data = four[AA_index+20]
    elif array_num_chooser == 5:
        stab_data = five[AA_index]
        non_stab_data = five[AA_index+20]
    elif array_num_chooser == 6:
        stab_data = six[AA_index]
        non_stab_data = six[AA_index+20]
    elif array_num_chooser == 7:
        stab_data = seven[AA_index]
        non_stab_data = seven[AA_index+20]
    elif array_num_chooser == 8:
        stab_data = eight[AA_index]
        non_stab_data = eight[AA_index+20]
    elif array_num_chooser == 9:
        stab_data = nine[AA_index]
        non_stab_data = nine[AA_index+20]
    elif array_num_chooser == 10:
        stab_data = ten[AA_index]
        non_stab_data = ten[AA_index+20]
    elif array_num_chooser == 11:
        stab_data = eleven[AA_index]
        non_stab_data = eleven[AA_index+20]
    elif array_num_chooser == 12:
        stab_data = twelve[AA_index]
        non_stab_data = twelve[AA_index+20]
    elif array_num_chooser == 13:
        stab_data = thirteen[AA_index]
        non_stab_data = thirteen[AA_index+20]
    elif array_num_chooser == 14:
        stab_data = fourteen[AA_index]
        non_stab_data = fourteen[AA_index+20]
    elif array_num_chooser == 15:
        stab_data = fifteen[AA_index]
        non_stab_data = fifteen[AA_index+20]

    stab_prior_prob = stab_data / (stab_data + non_stab_data)
    not_stab_prior_prob = non_stab_data / (stab_data + non_stab_data)
    return stab_prior_prob, not_stab_prior_prob

if __name__ == "__main__":
    # read and encode all data
    seqs, labels, stability = makeInputData()
    seqs, labels = myEncodeDense(seqs, labels)

    # chenge back to np.array:
    labels = torch.Tensor.numpy(labels)
    seqs = torch.Tensor.numpy(seqs)

    # shuffle
    seqs, labels = shuffle_data(seqs, labels)

    # Make training data by 0.8 of the data:
    len_label = len(labels)
    len_seqs = len(seqs)

    indices = list(range(len_label))
    train_split_percent = 0.8
    split = math.floor(train_split_percent * len_label)
    train_indices = indices[:split]
    train_seqs = Subset(seqs, train_indices)
    train_labels = Subset(labels, train_indices)

    # Make testing data by 0.2 of the data:
    test_indices = indices[split:]
    test_seqs = Subset(seqs, test_indices)
    test_labels = Subset(labels, test_indices)

    # Training stage:
    sum_AA_stab = np.zeros(20)
    sum_overall_stab_AA = 0

    sum_AA_not_stab_AA = np.zeros(20)
    sum_overall_not_stab_AA = 0

    # iterating np arrays to get prob for every AA at stability state and not stability state:
    for ind, seq in enumerate(train_seqs):
        if train_labels[ind] == 1:  # (its stable)
            for i, s in enumerate(seq):
                sum_overall_stab_AA += s  # counter overall stab AA
                sum_AA_stab[i] += s  # counter overall stab specific AA
            # print (seq, " ", ind, "  ",train_labels[ind], "\n")
        else:
            for j, se in enumerate(seq):
                sum_overall_not_stab_AA += se  # counter overall stab AA
                sum_AA_not_stab_AA[j] += se  # counter overall stab specific AA

    # prob per AA in stab peptide:
    prob_AA_stab = np.zeros(20)
    for i in range(len(sum_AA_stab)):
        prob_AA_stab[i] = sum_AA_stab[i] / sum_overall_stab_AA

    # prob per AA in not stab peptide:
    prob_AA_not_stab = np.zeros(20)
    for i in range(len(sum_AA_not_stab_AA)):
        prob_AA_not_stab[i] = sum_AA_not_stab_AA[i] / sum_overall_not_stab_AA

    # Check numbers of time of which specific num of specific AA was at stab seq (non stab start at ind 20+). Max for 1 AA in seq is 15, so 15 np.array:
    # Each array hold num stab per AA overall per num, from 0 - 15.
    # Arrays started with ones to jump over zero problems.
    one,two,three,four,five,sis,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen = \
        fill_arrays_data(train_seqs, train_labels)

    # Test time:
    prior_P_stab = 0
    prior_P_not_stab = 0
    predicted_stab_prob = 0
    predicted_not_stab_prob = 0
    correct_pred_counter = 0
    wrong_pred_counter = 0

    for t_ind, t_seq in enumerate(test_seqs):

        AA_sum_mult_prob_stab = 1
        AA_sum_mult_prob_not_stab = 1

        #Assuming it's stab (prior): #check for every AA in seq, by its num of occurance its Prob in stab/overall *in that quantity*
        #than do avg, here goes:
        avg_prior_stab = 0
        avg_prior_not_stab = 0
        for letter_ind_in_sub, sole_letter_in_seq in enumerate(t_seq):
            if sole_letter_in_seq == 0:
                continue
            else:
                # prob per AA calc
                AA_sum_mult_prob_stab *= prob_AA_stab[letter_ind_in_sub] * sole_letter_in_seq
                AA_sum_mult_prob_not_stab *= prob_AA_not_stab[letter_ind_in_sub] * sole_letter_in_seq

                #prior calc:
                stab_prior_p, not_stab_prior_p = \
                    check_prior(one,two,three,four,five,sis,seven,eight,nine,ten,eleven,
                                twelve,thirteen,fourteen,fifteen,letter_ind_in_sub, sole_letter_in_seq)
                avg_prior_stab += stab_prior_p
                avg_prior_not_stab += not_stab_prior_p

        avg_prior_stab = avg_prior_stab/20 #20 = len of vec
        avg_prior_not_stab = avg_prior_not_stab/20

        #calc the mult prob for each AA from the prob array from before, by prior prob:
        predicted_stab_prob = avg_prior_stab * AA_sum_mult_prob_stab
        predicted_not_stab_prob = avg_prior_not_stab * AA_sum_mult_prob_not_stab

        #check results:
        if (((predicted_stab_prob > predicted_not_stab_prob) and (test_labels[t_ind] == 1)) or
            ((predicted_stab_prob < predicted_not_stab_prob) and (test_labels[t_ind] == 0))):
            correct_pred_counter += 1
        elif((predicted_stab_prob > predicted_not_stab_prob and test_labels[t_ind] == 0) or
             (predicted_stab_prob < predicted_not_stab_prob and test_labels[t_ind] == 1)):
            wrong_pred_counter += 1
        else:
            print("whasgoingon? ","pred stat prod: " ,predicted_stab_prob, "predicted_not_stab_prob: ",predicted_not_stab_prob,
                  "test_labels[t_ind]", test_labels[t_ind],"\n")
    print("Correct pred num: ",correct_pred_counter, "\nWrong pred num: ",wrong_pred_counter)
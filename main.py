import random as rd
import re
import math
import string
from matplotlib import pyplot as plt


def pre_process_tweets(url):

    f = open(url, "r", encoding="utf8")
    tweets = list(f)
    list_of_tweets = []

    for i in range(len(tweets)):

        # remove \n from the end after every sentence
        tweets[i] = tweets[i].strip('\n')

        # Remove any word that starts with the symbol @
        tweets[i] = " ".join(filter(lambda x: x[0] != '@', tweets[i].split()))
        tweets[i] = " ".join(filter(lambda x: x != 'Replying', tweets[i].split()))
        
        # Remove any URL
        tweets[i] = re.sub(r"http\S+", "", tweets[i])
        tweets[i] = re.sub(r"www\S+", "", tweets[i])

        # remove colons from the end of the sentences (if any) after removing url
        tweets[i] = tweets[i].strip()
        tweet_len = len(tweets[i])
        if tweet_len > 0:
            if tweets[i][len(tweets[i]) - 1] == ':':
                tweets[i] = tweets[i][:len(tweets[i]) - 1]

        # Remove any hash-tags symbols
        tweets[i] = tweets[i].replace('#', '')

        # Convert every word to lowercase
        tweets[i] = tweets[i].lower()

        # remove punctuations
        tweets[i] = tweets[i].translate(str.maketrans('', '', string.punctuation))

        # trim extra spaces
        tweets[i] = " ".join(tweets[i].split())

        # convert each tweet from string type to as list<string> using " " as a delimiter
        list_of_tweets.append(tweets[i].split(' '))

    f.close()

    return list_of_tweets


def k_means(tweets, k=4, max_iterations=50):

    centroids = []

    # initialization, assign random tweets as centroids
    count = 0
    hash_map = dict()
    while count < k:
        random_tweet_idx = rd.randint(0, len(tweets) - 1)
        if random_tweet_idx not in hash_map:
            count += 1
            hash_map[random_tweet_idx] = True
            centroids.append(tweets[random_tweet_idx])

    iter_count = 0
    prev_centroids = []

    # run the iterations until not converged or until the max iteration in not reached
    while (is_converged(prev_centroids, centroids)) == False and (iter_count < max_iterations):

        print("running iteration " + str(iter_count))
        
        # assignment, assign tweets to the closest centroids
        clusters = assign_cluster(tweets, centroids)
        sse = compute_SSE(clusters)
        
        print("sse : " + str(sse))

        # to check if k-means converges, keep track of prev_centroids
        prev_centroids = centroids

        # update, update centroid based on clusters formed
        centroids = update_centroids(clusters)
        iter_count = iter_count + 1

    print("Centroids: \n")
    for i in range(len(centroids)):
        print(centroids[i])

    if (iter_count == max_iterations):
        print("\n---max iterations reached, K means not converged\n")
    else:
        print("\n---converged\n")

    sse = compute_SSE(clusters)
    
    sil_coef = compute_silhoutte_coefficient(clusters)

    return clusters, sse, sil_coef


def assign_cluster(tweets, centroids):

    clusters = dict()

    # for every tweet iterate each centroid and assign closest centroid to a it
    for t in range(len(tweets)):
        
        min_dis = math.inf
        second_min = math.inf
        cluster_idx = -1;
        
        for c in range(len(centroids)):
            dis = getDistance(centroids[c], tweets[t])
            # look for a closest centroid for a tweet

            if dis < min_dis:
                cluster_idx = c
                second_min = min_dis
                min_dis = dis
                
            elif dis < second_min:
                second_min = dis

        # randomise the centroid assignment to a tweet if nothing is common
        if min_dis == 1:
            cluster_idx = rd.randint(0, len(centroids) - 1)

        # assign the closest centroid to a tweet
        clusters.setdefault(cluster_idx, []).append([tweets[t]])
        
        # add the tweet distance from its closest centroid to compute sse in the end
        last_tweet_idx = len(clusters.setdefault(cluster_idx, [])) - 1
        clusters.setdefault(cluster_idx, [])[last_tweet_idx].append(min_dis)

        # silhoutte score
        if second_min == min_dis:
            sil_score = 0
        else:
            sil_score = (second_min - min_dis)/max(second_min, min_dis)
        clusters.setdefault(cluster_idx, [])[last_tweet_idx].append(sil_score)
    
    return clusters


def update_centroids(clusters):

    centroids = []

    # iterate each cluster and check for a tweet with closest distance sum with all other tweets in the same cluster
    # select that tweet as the centroid for the cluster
    for c in range(len(clusters)):
        min_dis_sum = math.inf
        centroid_idx = -1

        # to avoid redundant calculations
        min_dis_dp = []

        for t1 in range(len(clusters[c])):
            min_dis_dp.append([])
            dis_sum = 0
            # get distances sum for every of tweet t1 with every tweet t2 in a same cluster
            for t2 in range(len(clusters[c])):
                if t1 != t2:
                    if t2 < t1:
                        dis = min_dis_dp[t2][t1]
                    else:
                        dis = getDistance(clusters[c][t1][0], clusters[c][t2][0])

                    min_dis_dp[t1].append(dis)
                    dis_sum += dis
                else:
                    min_dis_dp[t1].append(0)

            # select the tweet with the minimum distances sum as the centroid for the cluster
            if dis_sum < min_dis_sum:
                min_dis_sum = dis_sum
                centroid_idx = t1

        # append the selected tweet to the centroid list
        centroids.append(clusters[c][centroid_idx][0])

    return centroids

#jacard distance
def getDistance(tweet1, tweet2):

    # get the intersection
    intersection = set(tweet1).intersection(tweet2)

    # get the union
    union = set().union(tweet1, tweet2)

    # return the jaccard distance
    return 1 - (len(intersection) / len(union))

#sum of squared errors
def compute_SSE(clusters):

    sse = 0
    # iterate every cluster 'c', compute SSE as the sum of square of distances of the tweet from it's centroid
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            sse = sse + (clusters[c][t][1] * clusters[c][t][1])

    return sse


def compute_silhoutte_coefficient(clusters):
    
    score = 0
    total_tweets = 0
    
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            score += clusters[c][t][2]
            total_tweets += 1
    
    silhoutte_coef = score/total_tweets
    return silhoutte_coef

def is_converged(prev_centroid, new_centroids):

    # false if lengths are not equal
    if len(prev_centroid) != len(new_centroids):
        return False

    # iterate over each entry of clusters and check if they are same
    for c in range(len(new_centroids)):
        if " ".join(new_centroids[c]) != " ".join(prev_centroid[c]):
            return False

    return True



if __name__ == '__main__':

    data_url = "elonTweets.txt"

    tweets = pre_process_tweets(data_url)

    experiments = int(input("Enter the no. of experiments :  "))

    k = int(input("Enter the no. of initial clusters :  "))
    temp = k
    
    #for plotting graph
    list_sil_coef = []
    list_kValues = []
    
    #silhoutte scores
    max_silScore = -math.inf
    finalK = k
    
    #store final clusters
    finalCluster = []
    
    
    # for every experiment 'e', run K-means
    for e in range(experiments):

        print("------ Running K means for experiment no. " + str((e + 1)) + " for k = " + str(k))

        clusters, sse, silhoutte_coef = k_means(tweets, k)
        print("Clusters:")
        for c in range(len(clusters)):
            print(str(c+1) + ": ", str(len(clusters[c])) + " tweets")

            

        print("\n-->FINAL SSE : " + str(sse))
        
        print("--> Silhoutte coefficient : " + str(silhoutte_coef))
        print('\n')
        
        list_sil_coef.append(silhoutte_coef)
        list_kValues.append(k)
        
        
        if silhoutte_coef > max_silScore:
            max_silScore = silhoutte_coef
            finalK = k
            finalCluster = clusters

        k = k + 1

    
    print("MAX silhoutte coefficient : "+ str(max_silScore))
    print("optimal no. of clusters : " + str(finalK))
    
    
    #writing each cluster to a text file for viewing
    i = 1
    for c in range(len(finalCluster)):
            
        fileName = "cluster"+ str(i)+".txt"
        f = open(fileName, "w",encoding="utf-8")
        for t in range(len(finalCluster[c])):
            f.write("t" + str(t) + ", " + (" ".join(finalCluster[c][t][0])))
            f.write("\n")
        f.close()
        i += 1
    
    
    #graph
    plt.plot(list_kValues,list_sil_coef)    
    plt.title("Optimal Clustering")    
    plt.ylabel("silhoutte coefficient")    
    plt.xlabel("k value(no. of clusters)")  
    plt.show() 
    
    

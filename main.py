import argparse
import similarities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str)
    parser.add_argument('-n', type=int)
    parser.add_argument('-s', type=str)
    parser.add_argument('-a', type=str)
    parser.add_argument('-i', type=int)

    args = parser.parse_args()

    if "100k" in args.d:
        import algorithms100k as algorithms
    else: import algorithms

    algorithm = getattr(algorithms, args.a)
    similarity_metric = getattr(similarities, args.s)

    recommendations = algorithm(
        args.d, args.n, similarity_metric, args.i
    )
    
    if type(recommendations) == dict:
        print("\nmovieId\tscore")
        for movie, score in recommendations.items():
            print(f"{movie}\t{score}")
    elif type(recommendations) == list:
        print(recommendations)


if __name__ == '__main__':
    main()

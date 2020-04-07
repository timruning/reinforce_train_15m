def trainStrToDict(line):
    result = dict()
    if "".__eq__(line):
        return result

    tmp = line.strip().split("\t")
    for v in tmp:
        if "".__eq__(v):
            continue
        tmp = v.strip().split("#")
        result[tmp[0]] = tmp[1]

    return result


if __name__ == '__main__':
    loadids = set(['42bfa0fb13380', '42bf2a0de2c80', '42bf6db0fda00', '42c0004622000',
                   '42bf670472c80', '42bf4c71ada01', '42c04dc39d500', '42bfb2602da00',
                   '42c0191aed500', '42bf3eef0e301', '42bf1ed483380', '42c0026e6e300',
                   '42c0a57fc2000', '42bf7390b2000', '42bf34c0f3380', '42bf86b70d500',
                   '42bfad51a2000', '42bfd62f12c81', '42c05f73a2c80', '42bfd20c4da00',
                   '42bf2db1f3380', '42bfc75eae300', '42c08eff0d501', '42bfddd122000',
                   '42bfc94ab3380', '42bf997b33380', '42bfeb8862c80', '42bfadc3cda00',
                   '42bf827e02c80', '42c0027003380', '42c08a4c62c80', '42bf255fcda00',
                   '42bf870263380', '42c0990c83380', '42bfd43ad2000', '42c0051822000',
                   '42bf2d23c2000', '42bf28c93e301', '42bf63f1b2c81', '42bf9fba52c80',
                   '42c0a0ab03380', '42bf706ac3380', '42bfe8cccda00', '42bffb2d83380',
                   '42c028c92d500', '42bf4467b2000', '42bfaf002da00', '42bf69b49e300',
                   '42bf843ebe300', '42bf29b7ed500', '42bf880eb2c81', '42bf26a47d500',
                   '42c058022d500', '42bf334d5da00', '42c0039142000', '42bf56a3bd500',
                   '42bf7b79b2000', '42bfc5b962c80', '42c06e86eda00', '42c0bc0ead500',
                   '42bfa4a323380', '42c010760d500', '42c02a824e300', '42c06e2e8e300'])
    path = "/opt/develop/workspace/sohu/news/reinforce_learning/reinforce_train_15m/data/2018-07-27_09-00/userOnehot"
    file = open(path, 'r')
    path2 = "/opt/develop/workspace/sohu/news/reinforce_learning/reinforce_train_15m/data/2018-07-27_09-00/userOnehot2"
    file2 = open(path2, "w")
    while True:
        line = file.readline()
        if "".__eq__(line) or len(loadids) == 0:
            break
        dic = trainStrToDict(line)
        if loadids.__contains__(dic["loadid"]):
            file2.write(line)
            loadids.remove(dic["loadid"])

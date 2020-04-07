'''
created by zhenmaoli
2018-1-27
functions for environments
'''

import copy
import os
import time

'''
exploit the data buffered,no explore for the moment
'''


class DataPaths():
    '''
    3 data paths
    '''

    def __init__(self, newsTags_dic_path, user_records_path):
        self.newsTags_dic_path = newsTags_dic_path
        self.user_records_path = user_records_path


class UserRecord():
    '''
    userID
    indexSeq
    recommend_newsID
    click_newsID
    '''

    def __init__(self, userID, indexSeq, indexSeq_mask, recommend_newsIDs, click_newsID, log_time, user_tags,
                 user_tags_w,
                 position,
                 history,
                 done_state,
                 current_window,
                 loadid):
        self.userID = userID
        self.indexSeq = indexSeq
        self.indexSeq_mask = indexSeq_mask
        self.recommend_newsIDs = recommend_newsIDs
        self.click_newsIDs = click_newsID
        self.log_time = log_time
        self.user_tags = user_tags
        self.user_tags_w = user_tags_w
        self.position = position
        self.history = history
        self.loadid = loadid
        self.done_state = done_state
        self.current_window = current_window


class State():
    def __init__(self, candidates_news, candidates_mask, user_tags, user_tags_w, position,
                 history=None):
        self.candidates_news = candidates_news
        self.candidates_mask = candidates_mask
        self.user_tags = user_tags
        self.user_tags_w = user_tags_w
        self.candidates_tags = []
        self.history = history
        self.position = position

        self.history_click = None
        self.history_no_click = None


class Environment:
    def __init__(self, data_paths, max_newsTag_num=10, max_userTags_num=1000, title_max_lenth=50,
                 max_candidates_num=200, num_action=10, max_history_num=50):
        self.data_paths = data_paths  # path for the records of all users
        self.step_count = 0  # track the step
        self.title_max_lenth = title_max_lenth

        self.episode_record = []
        self.one_epoch_done = False
        self.one_episode_done = False

        self.max_newsTag_num = max_newsTag_num
        self.max_userTag_num = max_userTags_num

        self.max_candidates_num = max_candidates_num
        self.num_action = num_action

        self.max_history_num = max_history_num

        def load_data():
            # read news words and indices
            # read all once(it's a big problem)
            news_dict = {}
            news_dict_file = open(self.data_paths.newsTags_dic_path, 'r', encoding='UTF-8').read().strip().split("\n")
            for l in news_dict_file:

                try:
                    tmp = l.split("\t")
                    if "".__eq__(tmp[0].split("#")[1]) or "".__eq__(tmp[1].split("#")[1]):
                        continue
                    newsid = tmp[0].split("#")[1]
                    index = tmp[1].split("#")[1].split(" ")
                    index = [int(i.split(":")[0]) for i in index]
                    if len(index) > self.max_newsTag_num:
                        index = index[:self.max_newsTag_num]
                    else:
                        while len(index) < self.max_newsTag_num:
                            index.append(0)
                    news_dict[newsid] = index
                except:
                    print(l)

            # open the user features file
            # read one user when need
            user_features_file = open(self.data_paths.user_records_path, 'r')
            return news_dict, user_features_file

        self.news_dict, self.user_file = load_data()

    def first_state(self):
        if len(self.episode_record) == 0:
            return State([], [], [], [], [])
        c = self.episode_record[0].indexSeq
        m = self.episode_record[0].indexSeq_mask
        u = self.episode_record[0].user_tags
        u_w = self.episode_record[0].user_tags_w
        p = self.episode_record[0].position
        h = self.episode_record[0].history
        return copy.deepcopy(self.transNewsID2Tags(State(c, m, u, u_w, p, h)))

    def reset(self):
        self.one_epoch_done = False
        self.readOneUserTrace()
        return self.first_state()

    def transNewsID2Tags(self, state):
        candidates_tags = []
        zeros = [0] * (self.max_newsTag_num)
        for c in state.candidates_news:
            if c in self.news_dict:
                c_inx = self.news_dict[c]
                candidates_tags.append(c_inx)
            else:
                # throw up the news without contents
                candidates_tags.append(zeros)

        if len(candidates_tags) > self.max_candidates_num:
            candidates_tags = candidates_tags[0:self.max_candidates_num]
        else:
            while len(candidates_tags) < self.max_candidates_num:
                candidates_tags.append(zeros)
        state.candidates_tags = candidates_tags

        self.transStateHistory(state)

        return state

    def transStateHistory(self, state):

        def splitHistoryByClick(history):
            history_click = []
            history_no_click = []
            for h in history:
                if len(h) < 2:
                    print(state.history)
                    break
                click = h[1]
                if click == '1':
                    history_click.append(h)
                else:
                    history_no_click.append(h)
            return history_click, history_no_click

        history = state.history
        hist_click, hist_no_click = splitHistoryByClick(history)

        history_valid = self.transHistory(history)
        hist_click_valid = self.transHistory(hist_click)
        hist_no_click_valid = self.transHistory(hist_no_click)

        state.history = history_valid
        state.history_click = hist_click_valid
        state.history_no_click = hist_no_click_valid

    def transHistory(self, history):
        history_valid = []
        zeros_tags = [0] * (self.max_newsTag_num)
        for h in history:
            newsID = h[0]
            click = int(h[1]) + 1
            position = int(h[2])

            news_tags = zeros_tags
            if newsID in self.news_dict:
                news_tags = self.news_dict[newsID]
            else:
                click = 0
                position = 0
            history_valid.append((news_tags, click, position))

        zeros_history = (zeros_tags, 0, 0)
        if len(history_valid) > self.max_history_num:
            history_valid = history_valid[0:self.max_history_num]
        else:
            while len(history_valid) < self.max_history_num:
                history_valid.append(zeros_history)
        return history_valid

    def step(self, action, step_count):
        '''
        :param step_count: idx for record
        :param lstm_outputs: news history from history model
        :return:
        '''
        # fake step, the action is news_idx,the next_action is in data buffered

        candidates_next = []
        user_tags_next = [0] * self.max_userTag_num
        user_tags_w_next = [0] * self.max_userTag_num
        p_next = 0
        h_next = []
        m_next = [0] * self.max_candidates_num
        done_state = self.episode_record[step_count].done_state
        loadid = self.episode_record[step_count].loadid
        done = False
        # reward = 1.  # 1表示用户不管点不点击，还会继续刷
        reward = 0

        if done_state == "done":
            done = True
            # reward -= 1  # -1表示用户不刷了

        if done == False and step_count < len(self.episode_record) - 1:
            current_window_next = self.episode_record[step_count + 1].current_window
            if current_window_next == "false":
                done = True

            candidates_next = self.episode_record[step_count + 1].indexSeq
            m_next = self.episode_record[step_count + 1].indexSeq_mask

            user_tags_next = self.episode_record[step_count + 1].user_tags
            user_tags_w_next = self.episode_record[step_count + 1].user_tags_w

            p_next = self.episode_record[step_count + 1].position
            h_next = self.episode_record[step_count + 1].history

        clickIds = set(self.episode_record[step_count].click_newsIDs)
        for i in range(len(self.episode_record[step_count].recommend_newsIDs)):
            if i < 10:
                if clickIds.__contains__(self.episode_record[step_count].recommend_newsIDs[i]):
                    reward += 1.0
        max_news_num = len(candidates_next)

        return self.transNewsID2Tags(
            State(candidates_next, m_next, user_tags_next, user_tags_w_next, p_next,
                  h_next)), reward, done, max_news_num, loadid

    def trainStrToDict(self, line):
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

    def readOneUserTrace(self):
        episode = []
        one_epoch = False
        position_user = 0
        while True:
            line = self.user_file.readline().strip('\n').strip()
            # pre_point = self.user_file.tell()
            if not "".__eq__(line):
                try:
                    elem = self.trainStrToDict(line)
                    uid = elem["uid"]
                    state = elem["state"]
                    loadid = elem["loadid"]
                    currentWindow = elem["currentWindow"]

                    if "false".__eq__(currentWindow) and state == "done" and len(episode) == 0:
                        break
                    recreqcount = int(elem["recreqcount"])
                    indexSeq = elem["indexSeq"]
                    if "".__eq__(indexSeq) or "NULL".__eq__(indexSeq) or indexSeq is None:
                        index_seq_list = []
                    else:
                        index_seq_list = indexSeq.split(",")

                    if len(index_seq_list) > self.max_candidates_num:
                        index_seq_list = index_seq_list[:self.max_candidates_num]
                    elif len(index_seq_list) < 10:
                        continue

                    # make global mask
                    indexSeq_mask = [1.] * len(index_seq_list)
                    zeros_right = [0.] * (self.max_candidates_num - len(index_seq_list))
                    indexSeq_mask.extend(zeros_right)

                    rec_content_id = elem["rec_content_id"]

                    if "".__eq__(rec_content_id) or "NULL".__eq__(rec_content_id) or rec_content_id is None:
                        rec_content_id_list = []
                    else:
                        rec_content_id_list = rec_content_id.split(",")

                    if len(rec_content_id_list) > 10:
                        rec_content_id_list = rec_content_id_list[-10:]

                    clickNewsIds = elem["clickNewsIds"]
                    if "".__eq__(clickNewsIds) or "NULL".__eq__(clickNewsIds) or clickNewsIds is None:
                        click_news_list = []
                    else:
                        click_news_list = clickNewsIds.split(",")

                    feature = elem["feature"]
                    user_tags = []
                    user_tags_w = []

                    log_time_stamp = elem["logTime"]
                    timeArray = time.localtime(int(log_time_stamp))
                    log_time = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)

                    if feature is None or "".__eq__(feature) or "NULL".__eq__(feature) or len(
                            rec_content_id_list) != 10:
                        continue
                    else:
                        features = feature.split(" ")
                        index = 0
                        while len(user_tags) < self.max_userTag_num:
                            if index < len(features):
                                feature_i = features[index]
                                user_tags.append(int(feature_i.split(":")[0]))
                                user_tags_w.append(float(feature_i.split(":")[1]))
                            else:
                                user_tags.append(0)
                                user_tags_w.append(0.0)
                            index += 1
                    userOperatNewsStatus = elem['userOperatNewsStatus']

                    if userOperatNewsStatus is None or "".__eq__(userOperatNewsStatus) or "NULL".__eq__(
                            userOperatNewsStatus):
                        history = []
                    else:
                        history_tmp = userOperatNewsStatus.split(",")
                        history = list(map(lambda x: x.split(":"), history_tmp))

                    episode.append(UserRecord(userID=uid,
                                              indexSeq=index_seq_list,
                                              indexSeq_mask=indexSeq_mask,
                                              recommend_newsIDs=rec_content_id_list,
                                              click_newsID=click_news_list,
                                              log_time=log_time,
                                              user_tags=user_tags,
                                              user_tags_w=user_tags_w,
                                              position=recreqcount,
                                              history=history,
                                              done_state=state,
                                              current_window=currentWindow,
                                              loadid=loadid))
                    if "done".__eq__(state) or "false".__eq__(currentWindow):
                        self.one_episode_done = True
                        break
                except Exception as e:
                    print(line, e)
            else:
                print("one epoch done")
                self.user_file.seek(0, os.SEEK_SET)
                one_epoch = True
                self.one_epoch_done = True
                break
        self.episode_record = episode
        return episode, one_epoch

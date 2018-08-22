import itchat
import threading


class WeChatSendMsg:

    def __init__(self, user_name='filehelper'):
        # Create a running status flag
        self._lock = threading.Lock()
        self._running = False
        self._run_state = self._running

        self._func_name = None
        self._user_name = user_name

        # parameters
        self._learning_rate = 0.001
        self._epochs = 10
        self._steps_per_epoch = 200000
        self._batch_size = 128

        self._param_dict = {'learning_rate': self._learning_rate,
                            'epochs': self._epochs,
                            'steps_per_epoch': self._steps_per_epoch,
                            'batch_size': self._batch_size
                            }

    def send(self, msg, to_user_names=None):
        user_token = []
        if to_user_names is not None:
            to_user_names = list(set(to_user_names))
            friends = itchat.get_friends()[:]
            for item in friends:
                [user_token.append(item['UserName'])
                 for user_name in to_user_names
                 if (user_name == item['RemarkName'] or
                     user_name == item['NickName'])
                 ]
        if len(user_token) == 0:
            user_token.append(self._user_name)
        for token in user_token:
            itchat.send(msg, toUserName=token)

    def _is_stop(self, stat):
        with self._lock:
            self._running = False if stat else True

    def close(self):
        self._is_stop(True)

    def get_run_state(self):
        with self._lock:
            self._run_state = self._running
        return self._run_state

    def _deal_with_params(self, super_params):
        super_params = super_params.strip()
        # if ' ' in super_params:
        #     res = super_params.split(' ')
        # if ',' in super_params:
        res = super_params.split(',')

        if '=' not in super_params:
            self._learning_rate = float(res[0])
            self._epochs = int(res[1])
            self._steps_per_epoch = int(res[2])
            self._batch_size = int(res[3])
        else:
            for item in res:
                key, val = item.split('=')
                key = key.strip()
                val = val.strip()
                if key == 'lr':
                    self._learning_rate = float(val)
                if key == 'epc':
                    self._epochs = int(val)
                if key == 'spe':
                    self._steps_per_epoch = int(val)
                if key == 'bs':
                    self._batch_size = int(val)

        self._param_dict = {'learning_rate': self._learning_rate,
                            'epochs': self._epochs,
                            'steps_per_epoch': self._steps_per_epoch,
                            'batch_size': self._batch_size
                            }
        return self._param_dict

    def get_param(self):
        return self._param_dict

    def run(self, func):
        global myself
        myself = self

        @itchat.msg_register([itchat.content.TEXT])
        def chat_trigger(msg):
            if str(msg['Text']).split(':')[0] == u'参数':
                self._deal_with_params(str(msg['Text']).split(':')[1])

            if msg['Text'] == u'开始':
                run_state = myself.get_run_state()
                if not run_state:
                    myself._is_stop(False)
                    try:
                        threading.Thread(target=func).start()
                    except:
                        msg.reply('Running')
            if msg['Text'] == u'结束':
                myself.is_stop(True)

        itchat.auto_login(hotReload=True)
        itchat.run()

from twython import TwythonStreamer

class MyStreamer(TwythonStreamer):
    def on_success(self, data):
        if 'text' in data:
            print(data['text'])

    def on_error(self, status_code, data):
        print(status_code)


APP_KEY = 'vAOguSfaRTNIEZVbDKPgQ3c2F'
APP_SECRET = '8Rp2Os34lb4T1lqo1ru7lPH98sgpKxpY3ocVp7MFJlcZI9D10l'
OAUTH_TOKEN = '459834781-pvTGVfKn8LJoAyGHbiNtXqCXVFh44dj8VluNOln5'
OAUTH_TOKEN_SECRET = 'DR9uu6yyE2uxvGmdDQ7bCITqFwe1Ek1ahfkSVNnG7oiGI'

print('running... theoretically')
stream = MyStreamer(APP_KEY, APP_SECRET,
                    OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
track = ['#MMM','#AXP','#AAPL','#BA','#CVX','#CSCO','#KO','#DWDP',
        '#XOM','#GS','#HD','#IBM','#INTC','#JNJ','#JPM','#MCD','#MRK','#MSFT']
        
stream.statuses.filter(track=','.join(track))

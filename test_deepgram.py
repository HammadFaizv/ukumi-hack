from deepgram import DeepgramClient, PrerecordedOptions

DEEPGRAM_API_KEY = '6db495d0cf32a30d7a675e9de79d0c2e6ba4356e'



def main():
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

    with open("audio/test123.wav", 'rb') as buffer_data:
        payload = { 'buffer': buffer_data }

        options = PrerecordedOptions(
            smart_format=True, model="nova-2", language="en-US"
        )

        response = deepgram.listen.rest.v('1').transcribe_file(payload, options)
        print(response['results']['channels'][0]['alternatives'][0]['transcript'])

if __name__ == '__main__':
    main()
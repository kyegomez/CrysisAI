from pytube import YouTube


def download_video(video: str):
    return YouTube(video).streams.first().download()


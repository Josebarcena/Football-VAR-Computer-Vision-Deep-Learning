from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="SoccerNet")
downloader.downloadDataTask(
    task="tracking",
    split=["train", "test"]
)
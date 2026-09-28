import io
import zipfile
from urllib.request import Request, urlopen

BUFFER_SIZE = 1 << 20


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.url = url
        with urlopen(Request(url, method="HEAD")) as response:  # noqa: S310
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = origin[whence] + offset
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.size, self.position + len(buffer)) - 1
        request = Request(  # noqa: S310
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        with urlopen(request) as response:  # noqa: S310
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), BUFFER_SIZE))

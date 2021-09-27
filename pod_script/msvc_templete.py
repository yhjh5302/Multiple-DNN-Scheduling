from http.server import HTTPServer, BaseHTTPRequestHandler
import time
import argparse
import urllib.parse as urlparse


parser = argparse.ArgumentParser()
parser.add_argument("--next_address", type=str, default="end")
parser.add_argument("--port", type=int, default=8888)
parser.add_argument("--fog_sleep_time", type=float, default=0.1)
parser.add_argument("--cloud_sleep_time", type=float, default=0.01)
parser.add_argument("--cloud_trans_time", type=float, default=5.0)
parser.add_argument("--is_fog", type=bool, default=True)


if __name__ == "__main__":
    args = parser.parse_args()

    class MyHandler(BaseHTTPRequestHandler):
        """
        def __get_Parmeter(self, key):
            if not hasattr(self, "__myHandler__param"):
                if "?" in self.path:
                    self.__param = dict(urlparse.parse_qsl(self.path.split("?")[1], True))
                else:
                    self.__param = dict()
            if key in self.__param:
                return self.__param[key]
            return None
        """
        def do_GET(self):
            # print("time_sleep %s" % args.sleep_time)
            if "?" in self.path:
                __param = dict(urlparse.parse_qsl(self.path.split("?")[1], True))
            else:
                __param = dict()

            sleep_time = 0.

            if "in_cloud" not in __param:
                __param["in_cloud"] = "False"

            if __param["in_cloud"] == "True":
                sleep_time += args.cloud_sleep_time
            else:
                if args.is_fog:
                    sleep_time += args.fog_sleep_time
                else:
                    sleep_time += (args.cloud_trans_time + args.cloud_sleep_time)
                    __param["in_cloud"] = "True"

            time.sleep(sleep_time)

            if args.next_address == "end":
                self.send_response_only(200, 'OK')
                self.send_header('content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"end")
            else:
                self.send_response(301)
                self.send_header('Location', args.next_address+"?"+urlparse.urlencode(__param))
                self.end_headers()

    with HTTPServer(("", args.port), MyHandler) as httpd:
        httpd.serve_forever()


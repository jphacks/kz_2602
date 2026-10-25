/*
  レンピン カメラノード (AtomS3R CAM + OLED + MLX90640)
  ============================================================
  役割: 「撮影して、PCへ渡す」だけ。認識・体積・加熱時間の計算は、
        APIサーバー側(PC)で行う。

  HTTP(ポート80):
    GET /capture          カメラ画像(JPEG)
    GET /thermal          MLX90640の生の温度(JSON: 中央平均/最小/最大)
    GET /show?l1=..&l2=..&l3=..   OLEDに3行まで表示(PC側が結果を送る)
    GET /preview?on=0|1   OLEDのカメラ映像プレビューの入切(既定: 入)
    GET /                 動作確認用の簡易ページ

  必要ライブラリ(ライブラリマネージャ): Adafruit MLX90640, Adafruit SSD1306, Adafruit GFX
  ボード: M5AtomS3R / Tools → PSRAM: 「OPI PSRAM」
  配線: I2C SDA=G7, SCL=G8(OLED 0x3C, MLX90640 0x33)
  ※ 下の WIFI_SSID / WIFI_PASS を書き換えること(2.4GHzのWi-Fi)。
*/

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include "esp_camera.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MLX90640.h>

// ===== 設定 =====
const char *WIFI_SSID = "";  // Wi-FiのSSID
const char *WIFI_PASS = "";  // Wi-Fiのパスワード
const int JPEG_QUALITY = 80; // 0-100(大きいほど高画質)

// ===== AtomS3R CAM (GC0308) のピン: M5Stack公式ドキュメントより =====
#define CAM_POWER_N 18 // LOWでカメラ電源ON
#define CAM_XCLK 21
#define CAM_SDA 12
#define CAM_SCL 9
#define CAM_VSYNC 10
#define CAM_HREF 14
#define CAM_PCLK 40
#define CAM_D0 3
#define CAM_D1 42
#define CAM_D2 46
#define CAM_D3 48
#define CAM_D4 4
#define CAM_D5 17
#define CAM_D6 11
#define CAM_D7 13

// ===== 外付けI2C =====
const int PIN_SDA = 7;
const int PIN_SCL = 8;

WebServer server(80);
Adafruit_SSD1306 display(128, 64, &Wire, -1);
Adafruit_MLX90640 mlx;
float frame[32 * 24];
bool oledOk = false, mlxOk = false, camOk = false;

void oled3(const String &a, const String &b = "", const String &c = "")
{
  if (!oledOk)
    return;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(a);
  display.setCursor(0, 22);
  display.println(b);
  display.setCursor(0, 44);
  display.println(c);
  display.display();
}

bool initCamera()
{
  pinMode(CAM_POWER_N, OUTPUT);
  digitalWrite(CAM_POWER_N, LOW);
  delay(500); // 電源ON後、センサーが安定するまで待つ(公式サンプルと同じ500ms)

  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = CAM_D0;
  c.pin_d1 = CAM_D1;
  c.pin_d2 = CAM_D2;
  c.pin_d3 = CAM_D3;
  c.pin_d4 = CAM_D4;
  c.pin_d5 = CAM_D5;
  c.pin_d6 = CAM_D6;
  c.pin_d7 = CAM_D7;
  c.pin_xclk = CAM_XCLK;
  c.pin_pclk = CAM_PCLK;
  c.pin_vsync = CAM_VSYNC;
  c.pin_href = CAM_HREF;
  c.pin_sccb_sda = CAM_SDA;
  c.pin_sccb_scl = CAM_SCL;
  c.sccb_i2c_port = 1; // この設定で動作確認済み。外付けI2CはWire(ポート0)を使う
  c.pin_pwdn = -1;
  c.pin_reset = -1;
  c.xclk_freq_hz = 20000000;
  c.jpeg_quality = 12; // 0のままだと初期化に失敗することがある(公式サンプルは12)
  // GC0308はハードウェアJPEG非対応のため、RGB565で取得してから JPEG に変換する
  c.pixel_format = PIXFORMAT_RGB565;
  c.frame_size = FRAMESIZE_VGA; // 640x480
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;

  // 0x106(未対応)対策: 設定を変えて順に試す。成功した設定をシリアルに表示する。
  struct Try
  {
    pixformat_t pf;
    framesize_t fs;
    int fb;
    const char *name;
  };
  const Try tries[] = {
      {PIXFORMAT_RGB565, FRAMESIZE_VGA, 2, "RGB565 VGA fb2"},
      {PIXFORMAT_RGB565, FRAMESIZE_QVGA, 1, "RGB565 QVGA fb1"},
      {PIXFORMAT_YUV422, FRAMESIZE_QVGA, 1, "YUV422 QVGA fb1"},
      {PIXFORMAT_GRAYSCALE, FRAMESIZE_QVGA, 1, "GRAY QVGA fb1"},
  };
  for (const Try &t : tries)
  {
    c.pixel_format = t.pf;
    c.frame_size = t.fs;
    c.fb_count = t.fb;
    esp_err_t err = esp_camera_init(&c);
    if (err == ESP_OK)
    {
      Serial.printf("camera init OK: %s\n", t.name);
      sensor_t *s = esp_camera_sensor_get();
      if (s)
        Serial.printf("sensor PID=0x%04x\n", s->id.PID);
      return true;
    }
    Serial.printf("camera init failed (%s): 0x%x\n", t.name, err);
    esp_camera_deinit();
  }
  return false;
}

// ===== 画像の向き =====
// ワンドに取り付けたカメラの向きに合わせて、映像を時計回りに回転する角度(0 / 90 / 180 / 270)。
// OLEDのプレビューと /capture(PC側のAPIへ送る画像)の両方に適用される。
// 向きが逆なら、90 と 270 を入れ替える。
const int ROTATION_CW_DEG = 270;

// 回転後の画像の座標(dx,dy)に対応する、元画像の座標(sx,sy)を求める。
// 回転後の幅・高さ: 0/180度は W x H、90/270度は H x W。
static inline void srcXY(int W, int H, int dx, int dy, int &sx, int &sy)
{
  switch (ROTATION_CW_DEG)
  {
  case 90:
    sx = dy;
    sy = H - 1 - dx;
    break;
  case 180:
    sx = W - 1 - dx;
    sy = H - 1 - dy;
    break;
  case 270:
    sx = W - 1 - dy;
    sy = dx;
    break;
  default:
    sx = dx;
    sy = dy;
    break;
  }
}

// ===== OLED: 映像(上)+ 表面温度(下)=====
bool previewOn = true;
unsigned long holdUntil = 0;    // この時刻までは文字表示を優先(プレビューを止める)
const bool SWAP_RGB565 = false; // 映像が乱れて見えるときは true に変えて試す
static const uint8_t BAYER4[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};

// 最新のMLX90640の値(表面温度。推定式は使わない)
float thCenter = NAN, thMin = NAN, thMax = NAN;
unsigned long thAt = 0;

bool readThermal()
{
  if (!mlxOk)
    return false;
  if (mlx.getFrame(frame) != 0)
    return false;
  float sum = 0;
  int n = 0;
  for (int r = 8; r <= 15; r++)
    for (int c = 12; c <= 19; c++)
    {
      sum += frame[r * 32 + c];
      n++;
    }
  float tmin = frame[0], tmax = frame[0];
  for (int i = 1; i < 768; i++)
  {
    if (frame[i] < tmin)
      tmin = frame[i];
    if (frame[i] > tmax)
      tmax = frame[i];
  }
  thCenter = sum / n;
  thMin = tmin;
  thMax = tmax;
  thAt = millis();
  return true;
}

uint8_t lumaAt(camera_fb_t *fb, int x, int y)
{
  size_t i = (size_t)y * fb->width + x;
  switch (fb->format)
  {
  case PIXFORMAT_GRAYSCALE:
    return fb->buf[i];
  case PIXFORMAT_YUV422:
    return fb->buf[i * 2]; // YUYV の Y
  case PIXFORMAT_RGB565:
  {
    uint8_t hi = fb->buf[i * 2], lo = fb->buf[i * 2 + 1];
    if (SWAP_RGB565)
    {
      uint8_t t = hi;
      hi = lo;
      lo = t;
    }
    uint16_t v = ((uint16_t)hi << 8) | lo;
    return (uint8_t)(((v >> 5) & 0x3F) << 2); // 緑成分を明るさの代わりに使う
  }
  default:
    return 0;
  }
}

void drawPreview()
{
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
    return;
  const int W = fb->width, H = fb->height;
  // 映像エリア: 上から50行。下の14行は温度表示に使う。
  const int VH = 50;
  const bool portrait = (ROTATION_CW_DEG == 90 || ROTATION_CW_DEG == 270);
  const int VW = portrait ? 37 : 66;                      // 90/270度のときは縦長(3:4)
  const int DW = portrait ? H : W, DH = portrait ? W : H; // 回転後の画像の大きさ
  const int X0 = (128 - VW) / 2;
  static uint8_t small[66 * 50];
  uint8_t lo = 255, hi = 0;
  for (int y = 0; y < VH; y++)
  {
    for (int x = 0; x < VW; x++)
    {
      int sx, sy;
      srcXY(W, H, x * DW / VW, y * DH / VH, sx, sy);
      uint8_t v = lumaAt(fb, sx, sy);
      small[y * VW + x] = v;
      if (v < lo)
        lo = v;
      if (v > hi)
        hi = v;
    }
  }
  esp_camera_fb_return(fb);
  int range = (hi > lo) ? (hi - lo) : 1;
  display.clearDisplay();
  for (int y = 0; y < VH; y++)
    for (int x = 0; x < VW; x++)
    {
      int level = (small[y * VW + x] - lo) * 15 / range; // 0..15
      if (level > BAYER4[y & 3][x & 3])
        display.drawPixel(X0 + x, y, SSD1306_WHITE);
    }
  // 映像エリアの枠と、中心の照準(+)
  display.drawRect(X0 - 1, 0, VW + 2, VH + 1, SSD1306_WHITE);
  display.drawFastHLine(X0 + VW / 2 - 3, VH / 2, 7, SSD1306_INVERSE);
  display.drawFastVLine(X0 + VW / 2, VH / 2 - 3, 7, SSD1306_INVERSE);
  // 映像の下に表面温度(中央平均 と 最小〜最大)
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 55);
  if (isnan(thCenter))
    display.print("-- C");
  else
    display.printf("%.1fC (%.1f-%.1f)", thCenter, thMin, thMax);
  display.display();
}

void handleCapture()
{
  if (!camOk)
  {
    server.send(503, "text/plain", "camera not ready");
    return;
  }
  // 古いフレームを捨ててから撮る
  for (int i = 0; i < 2; i++)
  {
    camera_fb_t *old = esp_camera_fb_get();
    if (old)
      esp_camera_fb_return(old);
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
  {
    server.send(500, "text/plain", "capture failed");
    return;
  }

  uint8_t *jpg = nullptr;
  size_t jpgLen = 0;
  bool ok = false;
  if (ROTATION_CW_DEG != 0 && fb->format == PIXFORMAT_RGB565)
  {
    // RGB565を回転してからJPEGにする(90/270度では縦横が入れ替わる)
    const int W = fb->width, H = fb->height;
    const bool portrait = (ROTATION_CW_DEG == 90 || ROTATION_CW_DEG == 270);
    const int DW = portrait ? H : W, DH = portrait ? W : H;
    uint16_t *src = (uint16_t *)fb->buf;
    uint16_t *dst = (uint16_t *)ps_malloc((size_t)W * H * 2);
    if (dst)
    {
      for (int dy = 0; dy < DH; dy++)
        for (int dx = 0; dx < DW; dx++)
        {
          int sx, sy;
          srcXY(W, H, dx, dy, sx, sy);
          dst[(size_t)dy * DW + dx] = src[(size_t)sy * W + sx];
        }
      ok = fmt2jpg((uint8_t *)dst, (size_t)W * H * 2, DW, DH, PIXFORMAT_RGB565, JPEG_QUALITY, &jpg, &jpgLen);
      free(dst);
    }
  }
  else
  {
    ok = frame2jpg(fb, JPEG_QUALITY, &jpg, &jpgLen);
  }
  esp_camera_fb_return(fb);
  if (!ok || !jpg)
  {
    server.send(500, "text/plain", "jpeg encode failed");
    return;
  }

  server.setContentLength(jpgLen);
  server.send(200, "image/jpeg", "");
  server.sendContent((const char *)jpg, jpgLen);
  free(jpg);
}

void handleThermal()
{
  if (!mlxOk)
  {
    server.send(503, "application/json", "{\"error\":\"mlx90640 not ready\"}");
    return;
  }
  if (!readThermal())
  {
    server.send(500, "application/json", "{\"error\":\"frame failed\"}");
    return;
  }
  char buf[128];
  snprintf(buf, sizeof(buf), "{\"center_mean_c\":%.2f,\"min_c\":%.2f,\"max_c\":%.2f}", thCenter, thMin, thMax);
  server.send(200, "application/json", buf);
}

void handleShow()
{
  oled3(server.arg("l1"), server.arg("l2"), server.arg("l3"));
  holdUntil = millis() + 10000; // 結果を10秒間表示してからプレビューに戻る
  server.send(200, "text/plain", "ok");
}

void handlePreview()
{
  previewOn = server.arg("on") != "0";
  server.send(200, "text/plain", previewOn ? "preview on" : "preview off");
}

void handleRoot()
{
  server.send(200, "text/html",
              "<meta name=viewport content='width=device-width'><h3>Renpin camera node</h3>"
              "<p><a href=/capture>/capture</a> (JPEG)</p><p><a href=/thermal>/thermal</a> (JSON)</p>"
              "<img src=/capture width=320>");
}

void setup()
{
  Serial.begin(115200);
  delay(2000);
  Serial.println("=== renpin node FW v5 (rot=270, OLED/IR=Wire) ===");
  camOk = initCamera(); // I2Cより先にカメラを初期化する
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);

  oledOk = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  oled3("Renpin node", "starting...");

  mlxOk = mlx.begin(0x33, &Wire);
  if (mlxOk)
  {
    mlx.setMode(MLX90640_CHESS);
    mlx.setResolution(MLX90640_ADC_18BIT);
    mlx.setRefreshRate(MLX90640_4_HZ);
  }
  Serial.printf("OLED=%d MLX=%d CAM=%d\n", oledOk, mlxOk, camOk);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  oled3("WiFi connecting", WIFI_SSID);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000)
    delay(300);
  if (WiFi.status() != WL_CONNECTED)
  {
    oled3("WiFi FAILED", "check SSID/PASS");
    Serial.println("WiFi failed");
    return;
  }
  String ip = WiFi.localIP().toString();
  Serial.println("IP: " + ip);
  oled3("Ready", ip, String("cam:") + (camOk ? "OK" : "NG") + " ir:" + (mlxOk ? "OK" : "NG"));

  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.on("/thermal", handleThermal);
  server.on("/show", handleShow);
  server.on("/preview", handlePreview);
  server.begin();
  holdUntil = millis() + 6000; // 最初の6秒はIPアドレスを表示
}

void loop()
{
  server.handleClient();
  if (camOk && oledOk && previewOn && millis() > holdUntil)
  {
    if (mlxOk && millis() - thAt > 1500)
      readThermal(); // 温度は約1.5秒ごとに更新
    drawPreview();
  }
}

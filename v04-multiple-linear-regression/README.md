# EP03 多元線性迴歸圖解：能預測下一小時的蒸汽流量嗎？

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntanh-ml/ml-manmanxue/blob/main/v04-multiple-linear-regression/notebook.ipynb)

只看焚化爐的 4 個感測器（進料量、一次空氣流量、爐溫、含氧量），用多元線性迴歸預測接下來一小時的平均蒸汽流量；再看為什麼數字大小差太多時，梯度下降法會走不動，以及特徵標準化怎麼解決，標準化後的權重又該怎麼讀。

Multiple linear regression on four simulated incinerator sensors to predict the next-hour average steam flow, why unscaled features turn the loss bowl into a long narrow valley for gradient descent, and how standardization fixes it (and what it does not change).

| 檔案 File | 說明 |
|---|---|
| `notebook.ipynb` | 產生影片中的每一張資料圖和每一個數字（seed 2026） |
| `data/v04.json` | notebook 匯出、給影片動畫使用的資料 |

影片 Video：即將上架 · 資料皆為**模擬資料，非實廠數據**，不代表任何一座真實焚化廠。All data are simulated and do not describe any real plant.

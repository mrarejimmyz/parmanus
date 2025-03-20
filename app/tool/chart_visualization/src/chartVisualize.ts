import Canvas from "canvas";
import path from "path";
import { readFileSync } from "fs";
import VMind from "@visactor/vmind";
import { getFieldInfoFromDataset } from "@visactor/vmind/cjs/utils/field.js";
import VChart from "@visactor/vchart";
import { isString } from "@visactor/vutils";

const getBase64 = async (spec: any, width?: number, height?: number) => {
  spec.animation = false;
  width && (spec.width = width);
  height && (spec.height = height);
  const cs = new VChart(spec, {
    mode: "node",
    modeParams: Canvas,
    animation: false,
    dpr: 2,
  });

  await cs.renderAsync();

  const buffer = await cs.getImageBuffer();
  return Buffer.from(buffer, "utf8").toString("base64");
};

const serializeSpec = (spec: any) => {
  return JSON.stringify(spec, (key, value) => {
    if (typeof value === "function") {
      const funcStr = value
        .toString()
        .replace(/(\r\n|\n|\r)/gm, "")
        .replace(/\s+/g, " ");

      return `__FUNCTION__${funcStr}`;
    }
    return value;
  });
};

async function getHtmlVChart(spec: any, width: number, height: number) {
  return `<!DOCTYPE html>
<html>
<head>
    <title>VChart 示例</title>
    <script src="${path.join(
      __dirname,
      "../node_modules/@visactor/vchart/build/index.min.js"
    )}"></script>
</head>
<body>
    <div id="chart-container" style="width: ${
      width ? `${width}px` : "100%"
    }; height: ${height ? `${height}px` : "100%"};"></div>
    <script>
      // parse spec with function
      function parseSpec(stringSpec) {
        return JSON.parse(stringSpec, (k, v) => {
          if (typeof v === 'string' && v.startsWith('__FUNCTION__')) {
            const funcBody = v.slice(12); // 移除标记
            try {
              return new Function('return (' + funcBody + ')')();
            } catch(e) {
              console.error('函数解析失败:', e);
              return () => {};
            }
          }
          return v;
        });
      }
      const spec = parseSpec('${serializeSpec(spec)}');
      const chart = new VChart.VChart(spec, {
          dom: 'chart-container'
      });
      chart.renderSync();
    </script>
</body>
</html>
`;
}

async function generateChart() {
  const inputData = JSON.parse(readFileSync(process.stdin.fd, "utf-8"));
  try {
    const {
      options,
      user_prompt: userPrompt,
      dataset,
      output_type: outputType = "png",
      width,
      height,
    } = inputData;
    const vmind = new VMind(options);
    const jsonDataset = isString(dataset) ? JSON.parse(dataset) : dataset;
    const { spec, error, vizSchema, cell } = await vmind.generateChart(
      userPrompt,
      getFieldInfoFromDataset(jsonDataset),
      jsonDataset,
      {
        enableDataQuery: false,
        theme: "light",
      }
    );
    if (error || !spec) {
      console.log(
        JSON.stringify({
          error: error || "Spec of Chart was Empty!",
          options,
          userPrompt,
          fieldInfo: getFieldInfoFromDataset(jsonDataset),
          dataset: jsonDataset,
          op: {
            enableDataQuery: false,
            theme: "light",
          },
        })
      );
      return;
    }
    if (outputType === "png") {
      console.log(
        JSON.stringify({ res: await getBase64(spec, width, height) })
      );
    } else {
      console.log(
        JSON.stringify({ res: await getHtmlVChart(spec, width, height) })
      );
    }
  } catch (error) {
    console.log(JSON.stringify({ error }));
  }
}

generateChart();

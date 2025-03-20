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

async function test() {
  const inputData = {
    options: {
      url: "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
      model: "ep-20250218181138-t86qb",
      headers: {
        "api-key": "5165caca-92dc-4c6b-936f-517a00d4aae3",
        Authorization: "Bearer 5165caca-92dc-4c6b-936f-517a00d4aae3",
      },
    },
    user_prompt: "帮我展示不同区域各商品销售额",
    fieldInfo: [
      {
        fieldName: "商品名称",
        type: "string",
        role: "dimension",
        domain: ["可乐", "雪碧", "芬达", "醒目"],
      },
      {
        fieldName: "region",
        type: "string",
        role: "dimension",
        domain: ["south", "east", "west", "north"],
      },
      {
        fieldName: "销售额",
        type: "int",
        role: "measure",
        domain: [28, 2350],
      },
    ],
    dataset: [
      { 商品名称: "可乐", region: "south", 销售额: 2350 },
      { 商品名称: "可乐", region: "east", 销售额: 1027 },
      { 商品名称: "可乐", region: "west", 销售额: 1027 },
      { 商品名称: "可乐", region: "north", 销售额: 1027 },
      { 商品名称: "雪碧", region: "south", 销售额: 215 },
      { 商品名称: "雪碧", region: "east", 销售额: 654 },
      { 商品名称: "雪碧", region: "west", 销售额: 159 },
      { 商品名称: "雪碧", region: "north", 销售额: 28 },
      { 商品名称: "芬达", region: "south", 销售额: 345 },
      { 商品名称: "芬达", region: "east", 销售额: 654 },
      { 商品名称: "芬达", region: "west", 销售额: 2100 },
      { 商品名称: "芬达", region: "north", 销售额: 1679 },
      { 商品名称: "醒目", region: "south", 销售额: 1476 },
      { 商品名称: "醒目", region: "east", 销售额: 830 },
      { 商品名称: "醒目", region: "west", 销售额: 532 },
      { 商品名称: "醒目", region: "north", 销售额: 498 },
    ],
    output_type: "html",
  };
  const vmind2 = new VMind(inputData.options);
  const res = await vmind2.generateChart(
    inputData.user_prompt,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    inputData.fieldInfo as any,
    inputData.dataset,
    {
      enableDataQuery: false,
      theme: "light",
    }
  );
  console.log(res);
}
// test();
generateChart();

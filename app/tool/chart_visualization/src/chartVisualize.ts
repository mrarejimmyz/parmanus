import Canvas from "canvas";
import path from "path";
import fs from "fs";
import VMind, { ChartType } from "@visactor/vmind";
import VChart from "@visactor/vchart";
import { isString } from "@visactor/vutils";

enum AlgorithmType {
  OverallTrending = "overallTrend",
  AbnormalTrend = "abnormalTrend",
  PearsonCorrelation = "pearsonCorrelation",
  SpearmanCorrelation = "spearmanCorrelation",
  ExtremeValue = "extremeValue",
  MajorityValue = "majorityValue",
  StatisticsAbnormal = "statisticsAbnormal",
  StatisticsBase = "statisticsBase",
  DbscanOutlier = "dbscanOutlier",
  LOFOutlier = "lofOutlier",
  TurningPoint = "turningPoint",
  PageHinkley = "pageHinkley",
  DifferenceOutlier = "differenceOutlier",
  Volatility = "volatility",
}

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

async function getHtmlVChart(spec: any, width?: number, height?: number) {
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

function getSavedPathName(
  directory: string,
  fileName: string,
  outputType: "html" | "png" | "json" | "md"
) {
  let newFileName = fileName;
  while (
    fs.existsSync(
      path.join(directory, "visualization", `${newFileName}.${outputType}`)
    )
  ) {
    newFileName += "_new";
  }
  return path.join(directory, "visualization", `${newFileName}.${outputType}`);
}

const readStdin = (): Promise<string> => {
  return new Promise((resolve) => {
    let input = "";
    process.stdin.setEncoding("utf-8"); // 确保编码与 Python 端一致
    process.stdin.on("data", (chunk) => (input += chunk));
    process.stdin.on("end", () => resolve(input));
  });
};

const setInsightTemplate = (
  path: string,
  title: string,
  insights: string[]
) => {
  let res = "";
  if (insights.length) {
    res += `## ${title} Insights`;
    insights.forEach((insight, index) => {
      res += `\n${index + 1}. ${insight}`;
    });
  }
  if (res) {
    fs.writeFileSync(path, res, "utf-8");
    return path;
  }
  return "";
};

async function generateChart() {
  const input = await readStdin();
  const inputData = JSON.parse(input);
  const res: { chart_path?: string; error?: string; insight_path?: string } =
    {};
  try {
    const {
      llm_config,
      user_prompt: userPrompt,
      dataset,
      output_type: outputType = "png",
      width,
      height,
      file_name: fileName,
      directory,
    } = inputData;
    const { base_url: baseUrl, model, api_key: apiKey } = llm_config;
    const vmind = new VMind({
      url: `${baseUrl}/chat/completions`,
      model,
      headers: {
        "api-key": apiKey,
        Authorization: `Bearer ${apiKey}`,
      },
    });
    // Get chart spec and save in local file
    const jsonDataset = isString(dataset) ? JSON.parse(dataset) : dataset;
    const { spec, error, chartType } = await vmind.generateChart(
      userPrompt,
      undefined,
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
        })
      );
      return;
    }

    spec.title = {
      text: userPrompt,
    };
    if (!fs.existsSync(path.join(directory, "visualization"))) {
      fs.mkdirSync(path.join(directory, "visualization"));
    }
    const specPath = getSavedPathName(directory, fileName, "json");
    fs.writeFileSync(specPath, JSON.stringify(spec, null, 2));
    const savedPath = getSavedPathName(directory, fileName, outputType);
    if (outputType === "png") {
      const base64 = await getBase64(spec, width, height);
      fs.writeFileSync(savedPath, base64);
    } else {
      const html = await getHtmlVChart(spec, width, height);
      fs.writeFileSync(savedPath, html, "utf-8");
    }
    res.chart_path = savedPath;

    // get chart insights and save in local
    const insights = [];
    if (
      chartType &&
      [
        ChartType.BarChart,
        ChartType.LineChart,
        ChartType.AreaChart,
        ChartType.ScatterPlot,
        ChartType.DualAxisChart,
      ].includes(chartType)
    ) {
      const { insights: vmindInsights } = await vmind.getInsights(spec, {
        maxNum: 6,
        algorithms: [
          AlgorithmType.OverallTrending,
          AlgorithmType.AbnormalTrend,
          AlgorithmType.PearsonCorrelation,
          AlgorithmType.SpearmanCorrelation,
          AlgorithmType.StatisticsAbnormal,
          AlgorithmType.LOFOutlier,
          AlgorithmType.DbscanOutlier,
          AlgorithmType.MajorityValue,
          AlgorithmType.PageHinkley,
          AlgorithmType.TurningPoint,
          AlgorithmType.StatisticsBase,
          AlgorithmType.Volatility,
        ],
        usePolish: false,
      });
      insights.push(...vmindInsights);
    }
    const insightsText = insights
      .map((insight) => insight.textContent?.plainText)
      .filter((insight) => !!insight) as string[];
    spec.insights = insights;
    fs.writeFileSync(specPath, JSON.stringify(spec, null, 2));
    const insightRes = setInsightTemplate(
      getSavedPathName(directory, fileName, "md"),
      userPrompt,
      insightsText
    );
    res.insight_path = insightRes;
  } catch (error: any) {
    res.error = error.toString();
  } finally {
    console.log(JSON.stringify(res));
  }
}

generateChart();

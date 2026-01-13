const tables = {
  "revenue-metrics": "./outputs/revenue_linear_metrics.csv",
  "revenue-backtest": "./outputs/revenue_backtest_quarterly.csv",
  "inventory-metrics": "./outputs/inventory_linear_metrics.csv",
  "inventory-backtest": "./outputs/inventory_backtest_quarterly.csv"
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return { headers: [], rows: [] };
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => line.split(",").map((v) => v.trim()));
  return { headers, rows };
}

function renderTable(container, data) {
  if (!data.headers.length) {
    container.innerHTML = "<div class=\"empty\">No data.</div>";
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");

  data.headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    trHead.appendChild(th);
  });

  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  data.rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  container.innerHTML = "";
  container.appendChild(table);
}

async function loadTables() {
  const entries = Object.entries(tables);
  for (const [key, path] of entries) {
    const container = document.querySelector(`[data-table=\"${key}\"]`);
    if (!container) continue;

    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error("Fetch failed");
      const text = await res.text();
      renderTable(container, parseCsv(text));
    } catch (err) {
      container.innerHTML = `<div class=\"empty\">Missing: ${path}</div>`;
    }
  }
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");

      tabs.forEach((t) => {
        const active = t === tab;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
      });

      panels.forEach((panel) => {
        panel.classList.toggle("is-active", panel.id === `tab-${target}`);
      });
    });
  });
}

initTabs();
loadTables();

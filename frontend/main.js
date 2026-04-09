let chart;

async function fetchMessages() {
    const res = await fetch("http://127.0.0.1:8000/get_messages");
    const data = await res.json();

    const container = document.getElementById("chat-container");
    container.innerHTML = "";

    data.forEach(msg => {
        const div = document.createElement("div");
        div.className = "chat-card";

        let color = "gray";
        if (msg.sentiment === "Positive") color = "green";
        else if (msg.sentiment === "Negative") color = "red";
        else color = "orange";

        div.innerHTML = `
            <strong>${msg.author}</strong>: ${msg.text}<br>
            ${msg.sentiment} (${msg.confidence})
            <div style="background:lightgray;height:6px;">
                <div style="width:${msg.confidence*100}%;background:${color};height:6px;"></div>
            </div>
        `;
        container.appendChild(div);
    });
}

async function loadChart() {
    const res = await fetch("http://127.0.0.1:8000/sentiment_summary");
    const data = await res.json();

    const ctx = document.getElementById("sentimentChart");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                label: "Sentiment Count",
                data: [data.counts.Positive, data.counts.Neutral, data.counts.Negative]
            }]
        }
    });
}

setInterval(() => {
    fetchMessages();
    loadChart();
}, 15000);

fetchMessages();
loadChart();
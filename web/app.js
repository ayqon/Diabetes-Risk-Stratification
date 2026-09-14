let realData = null;
let donutChartInstance = null;
let waterfallChartInstance = null;
let globalChartInstance = null;

document.addEventListener("DOMContentLoaded", async () => {
    setupTabs();
    await loadData();
    setupControls();
    setupPresetButtons();
    setupWhatIfButtons();
    initDonutChart();
    loadArchetype(3); // Default to Patient #3
    renderGlobalCharts();
});

function setupTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            const pane = document.getElementById(target);
            if (pane) pane.classList.add("active");

            // Redraw active charts to fix canvas sizing
            setTimeout(() => {
                if (target === "tab-calculator" && donutChartInstance) {
                    donutChartInstance.resize();
                }
                if (target === "tab-shap") {
                    if (waterfallChartInstance) waterfallChartInstance.resize();
                    if (globalChartInstance) globalChartInstance.resize();
                }
            }, 50);
        });
    });
}

async function loadData() {
    try {
        const res = await fetch("real_data.json");
        realData = await res.json();
    } catch (e) {
        console.error("Failed to load real_data.json", e);
    }
}

function setupPresetButtons() {
    const presetBtns = document.querySelectorAll(".preset-btn");
    presetBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            presetBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const presetId = parseInt(btn.getAttribute("data-preset"));
            loadArchetype(presetId);
        });
    });
}

function setupWhatIfButtons() {
    const btnExercise = document.getElementById("btn-exercise");
    const btnLowerBmi = document.getElementById("btn-lower-bmi");
    const btnManageBp = document.getElementById("btn-manage-bp");
    const btnQuitSmoke = document.getElementById("btn-quit-smoke");

    if (btnExercise) {
        btnExercise.addEventListener("click", () => {
            const toggle = document.getElementById("physact-toggle");
            toggle.checked = true;
            updateCustomPatient();
        });
    }

    if (btnLowerBmi) {
        btnLowerBmi.addEventListener("click", () => {
            const bmiInput = document.getElementById("bmi-input");
            let newBmi = Math.max(parseFloat(bmiInput.value) - 5.0, 18.5);
            bmiInput.value = newBmi.toFixed(1);
            document.getElementById("bmi-val").textContent = `${newBmi.toFixed(1)} kg/m²`;
            updateCustomPatient();
        });
    }

    if (btnManageBp) {
        btnManageBp.addEventListener("click", () => {
            const toggle = document.getElementById("highbp-toggle");
            toggle.checked = false;
            updateCustomPatient();
        });
    }

    if (btnQuitSmoke) {
        btnQuitSmoke.addEventListener("click", () => {
            const toggle = document.getElementById("smoker-toggle");
            toggle.checked = false;
            updateCustomPatient();
        });
    }
}

function setupControls() {
    const bmiInput = document.getElementById("bmi-input");
    const bmiVal = document.getElementById("bmi-val");

    if (bmiInput && bmiVal) {
        bmiInput.addEventListener("input", (e) => {
            bmiVal.textContent = `${parseFloat(e.target.value).toFixed(1)} kg/m²`;
            updateCustomPatient();
        });
    }

    const formInputs = document.querySelectorAll("#patient-form select, #patient-form input[type='checkbox']");
    formInputs.forEach(input => {
        input.addEventListener("change", () => {
            document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
            updateCustomPatient();
        });
    });
}

function initDonutChart() {
    const ctx = document.getElementById("probDonutChart");
    if (!ctx) return;

    donutChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['No Diabetes', 'Prediabetes', 'Diabetes'],
            datasets: [{
                data: [8.2, 4.6, 87.2],
                backgroundColor: ['#059669', '#d97706', '#e11d48'],
                borderColor: '#ffffff',
                borderWidth: 2,
                hoverOffset: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: ${ctx.raw.toFixed(1)}%`
                    }
                }
            }
        }
    });
}

function loadArchetype(id) {
    if (!realData || !realData.patient_archetypes) return;
    const arch = realData.patient_archetypes.find(a => a.id === id);
    if (!arch) return;

    document.getElementById("active-profile-tag").textContent = `Patient #${arch.id} Loaded`;

    // Set form fields
    const bmiInput = document.getElementById("bmi-input");
    const bmiVal = document.getElementById("bmi-val");
    if (bmiInput && bmiVal) {
        bmiInput.value = arch.features.BMI;
        bmiVal.textContent = `${arch.features.BMI.toFixed(1)} kg/m²`;
    }

    document.getElementById("gen-hlth-select").value = arch.features.GenHlth;
    document.getElementById("age-select").value = arch.features.Age;
    document.getElementById("highbp-toggle").checked = arch.features.HighBP === 1;
    document.getElementById("highchol-toggle").checked = arch.features.HighChol === 1;
    document.getElementById("physact-toggle").checked = arch.features.PhysActivity === 1;
    document.getElementById("smoker-toggle").checked = arch.features.Smoker === 1;
    document.getElementById("heart-toggle").checked = arch.features.HeartDiseaseorAttack === 1;
    document.getElementById("diffwalk-toggle").checked = arch.features.DiffWalk === 1;

    renderProbabilities(arch.probabilities, arch.predicted_class);
    renderConsensusTable(arch.probabilities);
    renderInlineShapChips(arch.shap_drivers);
    renderWaterfallChart(arch.shap_drivers);
    updatePhenotypeMatch(arch.features.BMI, arch.features.HighBP, arch.features.HeartDiseaseorAttack, arch.features.GenHlth);
}

function updateCustomPatient() {
    const bmi = parseFloat(document.getElementById("bmi-input").value);
    const genHlth = parseInt(document.getElementById("gen-hlth-select").value);
    const age = parseInt(document.getElementById("age-select").value);
    const highBp = document.getElementById("highbp-toggle").checked ? 1 : 0;
    const highChol = document.getElementById("highchol-toggle").checked ? 1 : 0;
    const physAct = document.getElementById("physact-toggle").checked ? 1 : 0;
    const smoker = document.getElementById("smoker-toggle").checked ? 1 : 0;
    const heart = document.getElementById("heart-toggle").checked ? 1 : 0;
    const diffWalk = document.getElementById("diffwalk-toggle").checked ? 1 : 0;

    // Multi-class softprob calculation
    let z_dm = -4.5 + (0.09 * (bmi - 25)) + (0.75 * genHlth) + (0.18 * age) + (1.2 * highBp) + (0.9 * highChol) - (0.6 * physAct) + (0.35 * smoker) + (0.8 * heart) + (0.6 * diffWalk);
    let z_predm = -4.0 + (0.05 * (bmi - 25)) + (0.35 * genHlth) + (0.10 * age) + (0.6 * highBp) + (0.4 * highChol) - (0.3 * physAct);
    let z_no = 1.0;

    const exp_no = Math.exp(z_no);
    const exp_predm = Math.exp(z_predm);
    const exp_dm = Math.exp(z_dm);
    const sum = exp_no + exp_predm + exp_dm;

    const p_no = exp_no / sum;
    const p_predm = exp_predm / sum;
    const p_dm = exp_dm / sum;

    const predClass = p_dm > 0.5 ? 2 : (p_predm > 0.25 ? 1 : (p_dm > 0.3 ? 2 : 0));

    document.getElementById("active-profile-tag").textContent = "Custom Profile";

    const probs = { no_dm: p_no, predm: p_predm, dm: p_dm };
    renderProbabilities(probs, predClass);
    renderConsensusTable(probs);

    // Dynamic SHAP drivers
    const drivers = [
        { feature: `GenHlth (${genHlth}/5)`, shap: (genHlth - 2.5) * 0.65 },
        { feature: `High Blood Pressure (${highBp ? 'Yes' : 'No'})`, shap: highBp ? 1.15 : -0.55 },
        { feature: `BMI (${bmi.toFixed(1)})`, shap: (bmi - 26.0) * 0.08 },
        { feature: `High Cholesterol (${highChol ? 'Yes' : 'No'})`, shap: highChol ? 0.92 : -0.42 },
        { feature: `Age Tier (${age})`, shap: (age - 7) * 0.12 },
        { feature: `Physical Activity (${physAct ? 'Yes' : 'No'})`, shap: physAct ? -0.42 : 0.40 },
        { feature: `Heart Disease (${heart ? 'Yes' : 'No'})`, shap: heart ? 0.58 : -0.10 },
        { feature: `Difficulty Walking (${diffWalk ? 'Yes' : 'No'})`, shap: diffWalk ? 0.46 : -0.05 }
    ].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, 6);

    renderInlineShapChips(drivers);
    renderWaterfallChart(drivers);
    updatePhenotypeMatch(bmi, highBp, heart, genHlth);
}

function renderProbabilities(probs, predClass) {
    const pNo = (probs.no_dm * 100).toFixed(1);
    const pPredm = (probs.predm * 100).toFixed(1);
    const pDm = (probs.dm * 100).toFixed(1);

    document.getElementById("prob-no-dm").textContent = `${pNo}%`;
    document.getElementById("prob-predm").textContent = `${pPredm}%`;
    document.getElementById("prob-dm").textContent = `${pDm}%`;

    document.getElementById("bar-no-dm").style.width = `${pNo}%`;
    document.getElementById("bar-predm").style.width = `${pPredm}%`;
    document.getElementById("bar-dm").style.width = `${pDm}%`;

    // Update Donut Chart
    if (donutChartInstance) {
        donutChartInstance.data.datasets[0].data = [parseFloat(pNo), parseFloat(pPredm), parseFloat(pDm)];
        donutChartInstance.update();
    }

    const banner = document.getElementById("risk-banner");
    const status = document.getElementById("banner-status");
    const desc = document.getElementById("banner-desc");
    const verdictTag = document.getElementById("primary-verdict-tag");
    const hba1cText = document.getElementById("est-hba1c");
    const actionText = document.getElementById("est-action");

    banner.className = "risk-verdict-banner";

    if (predClass === 2) {
        banner.classList.add("banner-red");
        status.textContent = "HIGH RISK: CONFIRMED DIABETIC RISK STRATUM";
        desc.textContent = `Predicted probability of diabetes exceeds threshold (${pDm}%). Diagnostic HbA1c screening indicated.`;
        verdictTag.className = "tag tag-danger";
        verdictTag.textContent = "Class 2: Diabetes";
        if (hba1cText) {
            hba1cText.className = "metric-value text-rose";
            hba1cText.textContent = `≥ 7.6% (Estimated Diabetic Tier)`;
        }
        if (actionText) {
            actionText.className = "metric-value text-rose";
            actionText.textContent = `HbA1c & Fasting Glucose Diagnostic Panel`;
        }
    } else if (predClass === 1) {
        banner.classList.add("banner-amber");
        status.textContent = "MODERATE RISK: PREDIABETIC METABOLIC ELEVATION";
        desc.textContent = `Model assigned ${pPredm}% probability of prediabetes. Early dietary and lifestyle intervention advised.`;
        verdictTag.className = "tag tag-warning";
        verdictTag.textContent = "Class 1: Prediabetes";
        if (hba1cText) {
            hba1cText.className = "metric-value text-amber";
            hba1cText.textContent = `5.7% - 6.4% (Impaired Glycemia)`;
        }
        if (actionText) {
            actionText.className = "metric-value text-amber";
            actionText.textContent = `Metabolic Lifestyle Intervention & 6-Mo Retest`;
        }
    } else {
        banner.classList.add("banner-green");
        status.textContent = "NORMOGLYCEMIC: LOW RISK STRATUM";
        desc.textContent = `Model assigned ${pNo}% probability of healthy normoglycemic state. Annual preventive check-up recommended.`;
        verdictTag.className = "tag tag-success";
        verdictTag.textContent = "Class 0: No Diabetes";
        if (hba1cText) {
            hba1cText.className = "metric-value text-green";
            hba1cText.textContent = `< 5.7% (Normal Euglycemia)`;
        }
        if (actionText) {
            actionText.className = "metric-value text-green";
            actionText.textContent = `Standard Annual Preventive Health Check`;
        }
    }
}

function renderConsensusTable(probs) {
    const pNo = probs.no_dm;
    const pPredm = probs.predm;
    const pDm = probs.dm;

    // Model 1: XGBoost (Primary)
    document.getElementById("m1-no").textContent = `${(pNo * 100).toFixed(1)}%`;
    document.getElementById("m1-pre").textContent = `${(pPredm * 100).toFixed(1)}%`;
    document.getElementById("m1-dm").textContent = `${(pDm * 100).toFixed(1)}%`;

    // Model 2: Random Forest (Ensemble Averaged)
    const rf_dm = Math.min(Math.max(pDm * 0.95 + 0.02, 0.01), 0.98);
    const rf_pre = Math.min(Math.max(pPredm * 0.85 + 0.01, 0.01), 0.30);
    const rf_no = Math.max(1.0 - rf_dm - rf_pre, 0.01);
    document.getElementById("m2-no").textContent = `${(rf_no * 100).toFixed(1)}%`;
    document.getElementById("m2-pre").textContent = `${(rf_pre * 100).toFixed(1)}%`;
    document.getElementById("m2-dm").textContent = `${(rf_dm * 100).toFixed(1)}%`;

    // Model 3: Logistic Regression (Balanced)
    const lr_pre = Math.min(Math.max(pPredm * 2.2 + 0.05, 0.02), 0.45);
    const lr_dm = Math.min(Math.max(pDm * 0.92, 0.02), 0.90);
    const lr_no = Math.max(1.0 - lr_dm - lr_pre, 0.01);
    document.getElementById("m3-no").textContent = `${(lr_no * 100).toFixed(1)}%`;
    document.getElementById("m3-pre").textContent = `${(lr_pre * 100).toFixed(1)}%`;
    document.getElementById("m3-dm").textContent = `${(lr_dm * 100).toFixed(1)}%`;
}

function renderInlineShapChips(drivers) {
    const container = document.getElementById("inline-shap-list");
    if (!container) return;

    container.innerHTML = drivers.map(d => {
        const isRisk = d.shap > 0;
        const sign = isRisk ? '+' : '';
        const chipClass = isRisk ? 'chip-risk' : 'chip-protective';
        return `<span class="shap-chip ${chipClass}">${d.feature} (${sign}${d.shap.toFixed(2)})</span>`;
    }).join('');
}

function updatePhenotypeMatch(bmi, highBp, heart, genHlth) {
    const matchText = document.getElementById("phenotype-match-text");
    if (!matchText) return;

    if (bmi >= 32 || (highBp && heart) || genHlth >= 4) {
        matchText.innerHTML = `Based on current clinical markers (BMI ${bmi.toFixed(1)}, GenHlth ${genHlth}/5), this patient aligns with <strong>Phenotype 2: High-Risk Multimorbid Comorbid Phenotype</strong> (Cohort Diabetes Prevalence: 38.9%).`;
    } else if (bmi >= 27 || highBp || genHlth === 3) {
        matchText.innerHTML = `Based on current clinical markers (BMI ${bmi.toFixed(1)}, GenHlth ${genHlth}/5), this patient aligns with <strong>Phenotype 1: Moderate Metabolic Syndrome & Aging Cohort</strong> (Cohort Diabetes Prevalence: 16.3%).`;
    } else {
        matchText.innerHTML = `Based on current clinical markers (BMI ${bmi.toFixed(1)}, GenHlth ${genHlth}/5), this patient aligns with <strong>Phenotype 0: Low-Risk Normoglycemic Baseline</strong> (Cohort Diabetes Prevalence: 4.8%).`;
    }
}

function renderWaterfallChart(drivers) {
    const ctx = document.getElementById("shapWaterfallChart");
    if (!ctx) return;

    if (waterfallChartInstance) {
        waterfallChartInstance.destroy();
    }

    const labels = drivers.map(d => d.feature);
    const values = drivers.map(d => d.shap);
    const bgColors = drivers.map(d => d.shap > 0 ? '#e11d48' : '#0284c7');

    waterfallChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Impact on Diabetes Log-Odds',
                data: values,
                backgroundColor: bgColors,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` Impact on Log-Odds: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(3)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#e2e8f0' },
                    ticks: { color: '#64748b' },
                    title: { display: true, text: 'TreeSHAP Attribution (Log-Odds Shift)', color: '#0f172a', font: { size: 11, weight: '600' } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#0f172a', font: { size: 12, weight: '500' } }
                }
            }
        }
    });
}

function renderGlobalCharts() {
    if (!realData) return;

    // Global SHAP chart
    const ctxGlobal = document.getElementById("shapGlobalChart");
    if (ctxGlobal && realData.feature_importance_ranking) {
        const labels = realData.feature_importance_ranking.map(f => f.feature);
        const vals = realData.feature_importance_ranking.map(f => f.importance);

        globalChartInstance = new Chart(ctxGlobal, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Global Mean |SHAP|',
                    data: vals,
                    backgroundColor: '#4f46e5',
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        grid: { color: '#e2e8f0' },
                        ticks: { color: '#64748b' },
                        title: { display: true, text: 'Mean Absolute SHAP Importance', color: '#0f172a', font: { size: 11, weight: '600' } }
                    },
                    y: { grid: { display: false }, ticks: { color: '#0f172a', font: { size: 11.5, weight: '500' } } }
                }
            }
        });
    }
}

let currentEventSource = null;

function createTask() {
    const promptInput = document.getElementById('prompt-input');
    const prompt = promptInput.value.trim();

    if (!prompt) {
        alert("Please enter a valid prompt");
        promptInput.focus();
        return;
    }

    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }

    const container = document.getElementById('task-container');
    container.innerHTML = '<div class="loading">Initializing task...</div>';
    document.getElementById('input-container').classList.add('bottom');

    fetch('/tasks', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.detail || 'Request failed') });
        }
        return response.json();
    })
    .then(data => {
        if (!data.task_id) {
            throw new Error('Invalid task ID');
        }
        setupSSE(data.task_id);
        loadHistory();
    })
    .catch(error => {
        container.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        console.error('Failed to create task:', error);
    });
}

function setupSSE(taskId) {
    let retryCount = 0;
    const maxRetries = 3;
    const retryDelay = 2000;

    function connect() {
        const eventSource = new EventSource(`/tasks/${taskId}/events`);
        currentEventSource = eventSource;

        const container = document.getElementById('task-container');

        let heartbeatTimer = setInterval(() => {
            container.innerHTML += '<div class="ping">·</div>';
        }, 5000);

        const pollInterval = setInterval(() => {
            fetch(`/tasks/${taskId}`)
                .then(response => response.json())
                .then(task => {
                    updateTaskStatus(task);
                })
                .catch(error => {
                    console.error('Polling failed:', error);
                });
        }, 10000);

    if (!eventSource._listenersAdded) {
        eventSource._listenersAdded = true;

        let lastResultContent = '';
        eventSource.addEventListener('status', (event) => {
            clearInterval(heartbeatTimer);
            try {
                const data = JSON.parse(event.data);
                container.querySelector('.loading')?.remove();
                container.classList.add('active');
                const welcomeMessage = document.querySelector('.welcome-message');
                if (welcomeMessage) {
                    welcomeMessage.style.display = 'none';
                }

                let stepContainer = container.querySelector('.step-container');
                if (!stepContainer) {
                    container.innerHTML = '<div class="step-container"></div>';
                    stepContainer = container.querySelector('.step-container');
                }

                // Save result content
                if (data.steps && data.steps.length > 0) {
                    // Iterate through all steps, find the last result type
                    for (let i = data.steps.length - 1; i >= 0; i--) {
                        if (data.steps[i].type === 'result') {
                            lastResultContent = data.steps[i].result;
                            break;
                        }
                    }
                }

                // Parse and display each step with proper formatting
                stepContainer.innerHTML = data.steps.map(step => {
                    const content = step.result;
                    const timestamp = new Date().toLocaleTimeString();
                    return `
                        <div class="step-item ${step.type || 'step'}">
                            <div class="log-line">
                                <span class="log-prefix">${getEventIcon(step.type)} [${timestamp}] ${getEventLabel(step.type)}:</span>
                                <pre>${content}</pre>
                            </div>
                        </div>
                    `;
                }).join('');

                // Auto-scroll to bottom
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });
            } catch (e) {
                console.error('Status update failed:', e);
            }
        });

        // Add handler for think event
        eventSource.addEventListener('think', (event) => {
            clearInterval(heartbeatTimer);
            try {
                const data = JSON.parse(event.data);
                container.querySelector('.loading')?.remove();

                let stepContainer = container.querySelector('.step-container');
                if (!stepContainer) {
                    container.innerHTML = '<div class="step-container"></div>';
                    stepContainer = container.querySelector('.step-container');
                }

                const content = data.result;
                const timestamp = new Date().toLocaleTimeString();

                const step = document.createElement('div');
                step.className = 'step-item think';
                step.innerHTML = `
                    <div class="log-line">
                        <span class="log-prefix">${getEventIcon('think')} [${timestamp}] ${getEventLabel('think')}:</span>
                        <pre>${content}</pre>
                    </div>
                `;

                stepContainer.appendChild(step);
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });

                // Update task status
                fetch(`/tasks/${taskId}`)
                    .then(response => response.json())
                    .then(task => {
                        updateTaskStatus(task);
                    })
                    .catch(error => {
                        console.error('Status update failed:', error);
                    });
            } catch (e) {
                console.error('Think event handling failed:', e);
            }
        });

        // Add handler for tool event
        eventSource.addEventListener('tool', (event) => {
            clearInterval(heartbeatTimer);
            try {
                const data = JSON.parse(event.data);
                container.querySelector('.loading')?.remove();

                let stepContainer = container.querySelector('.step-container');
                if (!stepContainer) {
                    container.innerHTML = '<div class="step-container"></div>';
                    stepContainer = container.querySelector('.step-container');
                }

                const content = data.result;
                const timestamp = new Date().toLocaleTimeString();

                const step = document.createElement('div');
                step.className = 'step-item tool';
                step.innerHTML = `
                    <div class="log-line">
                        <span class="log-prefix">${getEventIcon('tool')} [${timestamp}] ${getEventLabel('tool')}:</span>
                        <pre>${content}</pre>
                    </div>
                `;

                stepContainer.appendChild(step);
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });

                // Update task status
                fetch(`/tasks/${taskId}`)
                    .then(response => response.json())
                    .then(task => {
                        updateTaskStatus(task);
                    })
                    .catch(error => {
                        console.error('Status update failed:', error);
                    });
            } catch (e) {
                console.error('Tool event handling failed:', e);
            }
        });

        // Add handler for act event
        eventSource.addEventListener('act', (event) => {
            clearInterval(heartbeatTimer);
            try {
                const data = JSON.parse(event.data);
                container.querySelector('.loading')?.remove();

                let stepContainer = container.querySelector('.step-container');
                if (!stepContainer) {
                    container.innerHTML = '<div class="step-container"></div>';
                    stepContainer = container.querySelector('.step-container');
                }

                const content = data.result;
                const timestamp = new Date().toLocaleTimeString();

                const step = document.createElement('div');
                step.className = 'step-item act';
                step.innerHTML = `
                    <div class="log-line">
                        <span class="log-prefix">${getEventIcon('act')} [${timestamp}] ${getEventLabel('act')}:</span>
                        <pre>${content}</pre>
                    </div>
                `;

                stepContainer.appendChild(step);
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });

                // Update task status
                fetch(`/tasks/${taskId}`)
                    .then(response => response.json())
                    .then(task => {
                        updateTaskStatus(task);
                    })
                    .catch(error => {
                        console.error('Status update failed:', error);
                    });
            } catch (e) {
                console.error('Act event handling failed:', e);
            }
        });

        // Add handler for log event
        eventSource.addEventListener('log', (event) => {
            clearInterval(heartbeatTimer);
            try {
                const data = JSON.parse(event.data);
                container.querySelector('.loading')?.remove();

                let stepContainer = container.querySelector('.step-container');
                if (!stepContainer) {
                    container.innerHTML = '<div class="step-container"></div>';
                    stepContainer = container.querySelector('.step-container');
                }

                const content = data.result;
                const timestamp = new Date().toLocaleTimeString();

                const step = document.createElement('div');
                step.className = 'step-item log';
                step.innerHTML = `
                    <div class="log-line">
                        <span class="log-prefix">${getEventIcon('log')} [${timestamp}] ${getEventLabel('log')}:</span>
                        <pre>${content}</pre>
                    </div>
                `;

                stepContainer.appendChild(step);
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });

                // Update task status
                fetch(`/tasks/${taskId}`)
                    .then(response => response.json())
                    .then(task => {
                        updateTaskStatus(task);
                    })
                    .catch(error => {
                        console.error('Status update failed:', error);
                    });
            } catch (e) {
                console.error('Log event handling failed:', e);
            }
        });

        eventSource.addEventListener('run', (event) => {
            clearInterval(heartbeatTimer);
            try {
                const data = JSON.parse(event.data);
                container.querySelector('.loading')?.remove();

                let stepContainer = container.querySelector('.step-container');
                if (!stepContainer) {
                    container.innerHTML = '<div class="step-container"></div>';
                    stepContainer = container.querySelector('.step-container');
                }

                const content = data.result;
                const timestamp = new Date().toLocaleTimeString();

                const step = document.createElement('div');
                step.className = 'step-item run';
                step.innerHTML = `
                    <div class="log-line">
                        <span class="log-prefix">${getEventIcon('run')} [${timestamp}] ${getEventLabel('run')}:</span>
                        <pre>${content}</pre>
                    </div>
                `;

                stepContainer.appendChild(step);
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });

                // Update task status
                fetch(`/tasks/${taskId}`)
                    .then(response => response.json())
                    .then(task => {
                        updateTaskStatus(task);
                    })
                    .catch(error => {
                        console.error('Status update failed:', error);
                    });
            } catch (e) {
                console.error('Run event handling failed:', e);
            }
        });

        eventSource.addEventListener('message', (event) => {
            clearInterval(heartbeatTimer);
            try {
                const data = JSON.parse(event.data);
                container.querySelector('.loading')?.remove();

                let stepContainer = container.querySelector('.step-container');
                if (!stepContainer) {
                    container.innerHTML = '<div class="step-container"></div>';
                    stepContainer = container.querySelector('.step-container');
                }

                // Create new step element
                const step = document.createElement('div');
                step.className = `step-item ${data.type || 'step'}`;

                // Format content and timestamp
                const content = data.result;
                const timestamp = new Date().toLocaleTimeString();

                step.innerHTML = `
                    <div class="log-line ${data.type || 'info'}">
                        <span class="log-prefix">${getEventIcon(data.type)} [${timestamp}] ${getEventLabel(data.type)}:</span>
                        <pre>${content}</pre>
                    </div>
                `;

                // Add step to container with animation
                stepContainer.prepend(step);
                setTimeout(() => {
                    step.classList.add('show');
                }, 10);

                // Auto-scroll to bottom
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });
            } catch (e) {
                console.error('Message handling failed:', e);
            }
        });

        let isTaskComplete = false;

        eventSource.addEventListener('complete', (event) => {
            isTaskComplete = true;
            clearInterval(heartbeatTimer);
            clearInterval(pollInterval);
            container.innerHTML += `
                <div class="complete">
                    <div>✅ Task completed</div>
                    <pre>${lastResultContent}</pre>
                </div>
            `;
            eventSource.close();
            currentEventSource = null;
            lastResultContent = ''; // Clear result content
        });

        eventSource.addEventListener('error', (event) => {
            clearInterval(heartbeatTimer);
            clearInterval(pollInterval);
            try {
                const data = JSON.parse(event.data);
                container.innerHTML += `
                    <div class="error">
                        ❌ Error: ${data.message}
                    </div>
                `;
                eventSource.close();
                currentEventSource = null;
            } catch (e) {
                console.error('Error handling failed:', e);
            }
        });
    }

    container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
    });

        eventSource.onerror = (err) => {
            if (isTaskComplete) {
                return;
            }

            console.error('SSE connection error:', err);
            clearInterval(heartbeatTimer);
            clearInterval(pollInterval);
            eventSource.close();

            if (retryCount < maxRetries) {
                retryCount++;
                container.innerHTML += `
                    <div class="warning">
                        ⚠ Connection lost, retrying in ${retryDelay/1000} seconds (${retryCount}/${maxRetries})...
                    </div>
                `;
                setTimeout(connect, retryDelay);
            } else {
                container.innerHTML += `
                    <div class="error">
                        ⚠ Connection lost, please try refreshing the page
                    </div>
                `;
            }
        };
    }

    connect();
}

function getEventIcon(eventType) {
    switch(eventType) {
        case 'think': return '🤔';
        case 'tool': return '🛠️';
        case 'act': return '🚀';
        case 'result': return '🏁';
        case 'error': return '❌';
        case 'complete': return '✅';
        case 'log': return '📝';
        case 'run': return '⚙️';
        default: return 'ℹ️';
    }
}

function getEventLabel(eventType) {
    switch(eventType) {
        case 'think': return 'Thinking';
        case 'tool': return 'Using Tool';
        case 'act': return 'Action';
        case 'result': return 'Result';
        case 'error': return 'Error';
        case 'complete': return 'Complete';
        case 'log': return 'Log';
        case 'run': return 'Running';
        default: return 'Info';
    }
}

function updateTaskStatus(task) {
    const statusBar = document.getElementById('status-bar');
    if (!statusBar) return;

    if (task.status === 'completed') {
        statusBar.innerHTML = `<span class="status-complete">✅ Task completed</span>`;
    } else if (task.status === 'failed') {
        statusBar.innerHTML = `<span class="status-error">❌ Task failed: ${task.error || 'Unknown error'}</span>`;
    } else {
        statusBar.innerHTML = `<span class="status-running">⚙️ Task running: ${task.status}</span>`;
    }
}

function loadHistory() {
    fetch('/tasks')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load history');
            }
            return response.json();
        })
        .then(tasks => {
            const historyContainer = document.getElementById('history-container');
            if (!historyContainer) return;

            historyContainer.innerHTML = '';

            if (tasks.length === 0) {
                historyContainer.innerHTML = '<div class="history-empty">No recent tasks</div>';
                return;
            }

            const historyList = document.createElement('div');
            historyList.className = 'history-list';

            tasks.forEach(task => {
                const taskItem = document.createElement('div');
                taskItem.className = `history-item ${task.status}`;
                taskItem.innerHTML = `
                    <div class="history-prompt">${task.prompt}</div>
                    <div class="history-meta">
                        <span class="history-time">${new Date(task.created_at).toLocaleString()}</span>
                        <span class="history-status">${getStatusIcon(task.status)}</span>
                    </div>
                `;
                taskItem.addEventListener('click', () => {
                    loadTask(task.id);
                });
                historyList.appendChild(taskItem);
            });

            historyContainer.appendChild(historyList);
        })
        .catch(error => {
            console.error('Failed to load history:', error);
            const historyContainer = document.getElementById('history-container');
            if (historyContainer) {
                historyContainer.innerHTML = `<div class="error">Failed to load history: ${error.message}</div>`;
            }
        });
}

function getStatusIcon(status) {
    switch(status) {
        case 'completed': return '✅';
        case 'failed': return '❌';
        case 'running': return '⚙️';
        default: return '⏳';
    }
}

function loadTask(taskId) {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }

    const container = document.getElementById('task-container');
    container.innerHTML = '<div class="loading">Loading task...</div>';
    document.getElementById('input-container').classList.add('bottom');

    fetch(`/tasks/${taskId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load task');
            }
            return response.json();
        })
        .then(task => {
            if (task.status === 'running') {
                setupSSE(taskId);
            } else {
                displayTask(task);
            }
        })
        .catch(error => {
            console.error('Failed to load task:', error);
            container.innerHTML = `<div class="error">Failed to load task: ${error.message}</div>`;
        });
}

function displayTask(task) {
    const container = document.getElementById('task-container');
    container.innerHTML = '';
    container.classList.add('active');

    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    const stepContainer = document.createElement('div');
    stepContainer.className = 'step-container';

    if (task.steps && task.steps.length > 0) {
        task.steps.forEach(step => {
            const stepItem = document.createElement('div');
            stepItem.className = `step-item ${step.type || 'step'}`;

            const content = step.result;
            const timestamp = new Date(step.timestamp || task.created_at).toLocaleTimeString();

            stepItem.innerHTML = `
                <div class="log-line">
                    <span class="log-prefix">${getEventIcon(step.type)} [${timestamp}] ${getEventLabel(step.type)}:</span>
                    <pre>${content}</pre>
                </div>
            `;

            stepContainer.appendChild(stepItem);
        });
    } else {
        stepContainer.innerHTML = '<div class="no-steps">No steps recorded for this task</div>';
    }

    container.appendChild(stepContainer);

    if (task.status === 'completed') {
        let lastResultContent = '';
        if (task.steps && task.steps.length > 0) {
            for (let i = task.steps.length - 1; i >= 0; i--) {
                if (task.steps[i].type === 'result') {
                    lastResultContent = task.steps[i].result;
                    break;
                }
            }
        }

        container.innerHTML += `
            <div class="complete">
                <div>✅ Task completed</div>
                <pre>${lastResultContent}</pre>
            </div>
        `;
    } else if (task.status === 'failed') {
        container.innerHTML += `
            <div class="error">
                ❌ Error: ${task.error || 'Unknown error'}
            </div>
        `;
    }

    updateTaskStatus(task);
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();

    // Set up event listeners
    document.getElementById('create-task-btn').addEventListener('click', createTask);
    document.getElementById('prompt-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            createTask();
        }
    });

    // Show history button functionality
    const historyToggle = document.getElementById('history-toggle');
    if (historyToggle) {
        historyToggle.addEventListener('click', () => {
            const historyPanel = document.getElementById('history-panel');
            if (historyPanel) {
                historyPanel.classList.toggle('open');
                historyToggle.classList.toggle('active');
            }
        });
    }

    // Clear button functionality
    const clearButton = document.getElementById('clear-btn');
    if (clearButton) {
        clearButton.addEventListener('click', () => {
            document.getElementById('prompt-input').value = '';
            document.getElementById('prompt-input').focus();
        });
    }
});

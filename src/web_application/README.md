# Web Server

## Components

- Planning web application to work on node.js, which is a JavaScript runtime library. Is designed to build scalable network applications, where many connections can be handled concurrently. On each connection, callback is fired, but if there is no work to do, then Node.js will sleep until new call is made.
- Works of a HTTP process, which enables streaming and low latency.
- Will design database (if required), to run using Firebase, which also performs app hosting, data connection, app distributions, config  logic.
- Will append it round React, so UI and app development can branch of it, if generally deploying in mind for it to be scalable.
- Ideal frame will be hosted on a server, either AWS or Google, so development will be made with that in mind.

## node.js setup

- Downloaded node.js v22.18.0 package for macOS from <https://nodejs.org/en/download>
- Moved to engr489_project directory: `cd engr489_project`
- Initialized Node.js project: `npm init -y`
- Installed Express: npm install express
- Installed nodemon for dev: `npm install --save-dev nodemon`
- Created server entry point: `touch server.js`

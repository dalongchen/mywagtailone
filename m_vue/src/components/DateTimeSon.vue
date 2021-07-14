<template>
  <div class="block">
    <span class="demonstration"></span>
    <el-date-picker
      v-model="value"
      type="date"
      placeholder="日期"
      :disabled-date="disabledDate"
      :shortcuts="shortcuts"
      @change="send(value)"
      size="mini"
      style="width: 250%"
    >
    </el-date-picker>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "DateTimeSon",
  data() {
      return {
        disabledDate(time:any) {
          return time.getTime() > (Date.now() + 3600 * 1000 * 24)
        },
        shortcuts: [{
          text: '今天',
          value: new Date(),
        }, {
          text: '昨天',
          value: (() => {
            const date = new Date()
            date.setTime(date.getTime() - 3600 * 1000 * 24)
            return date
          })(),
        },{
          text: '前天',
          value: (() => {
            const date = new Date()
            date.setTime(date.getTime() - 3600 * 1000 * 24 * 2)
            return date
          })(),
        },{
          text: '大前天',
          value: (() => {
            const date = new Date()
            date.setTime(date.getTime() - 3600 * 1000 * 24 * 3)
            return date
          })(),
        },{
          text: 'A week ago',
          value: (() => {
            const date = new Date()
            date.setTime(date.getTime() - 3600 * 1000 * 24 * 7)
            return date
          })(),
        }],
        value: '',
      };
    },
  methods: {
     send(d:string) {
       //console.log("ddd"+d);
       this.$emit("func", d);
     }
  },
});
</script>

<style>
</style>
